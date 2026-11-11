"""Reconstruct alternative vertex and receiver raster conventions from H and TX."""
import argparse, json, math, time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import Delaunay

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from src.cc_core import gpu_guard, atomic_json
from src.c69_registered import RegisteredTransport
from src.c67_physics import quadrature

A = R / 'audit/c79_receiver'
VERTEX = ['native','floor1','round1','round025','u8','u10','u12','smooth','sharpen','unwarp','warp','stagger','coarse2','anti_native']
QUERY = ['centroid','vertices','integer','half_after','half_before','xminus','xplus','yminus','yplus',
         'inner_x_vertex_y','vertex_x_inner_y','sharp','triangle_nearest','consistent_ties_flip']
COL = ['direct_lo','direct_hi','total_lo','total_hi','native_db_lo','native_db_hi','visible',
       'weak_face','shadow_area','log_cos_min','edge_on','weighted_weak_lo','weighted_weak_hi',
       'slope_x','slope_y','fractional_weight','min_cos','mixed_sign']

def warp(h, x, y):
    g = torch.stack([x /127.5-1, y/127.5-1], -1)[None]
    return F.grid_sample(h[None,None],g,mode='bilinear',align_corners=True,padding_mode='border')[0,0]

def queries(diag):
    from src.c66_lattice import Mesh
    mesh=Mesh(1,diag)
    centers=mesh.xy[mesh.faces].mean(1)
    tri=Delaunay(centers)
    u=np.linspace(centers[:,0].min(),centers[:,0].max(),256)
    v=np.linspace(-128,128,256)
    raw=np.arange(256)-128.
    specs={'centroid':(u,u), 'vertices':(v,v), 'integer':(raw,raw+1),
           'half_after':(u+.5,u+.5), 'half_before':(u-.5,u-.5), 'xminus':(u-.5,u),
           'xplus':(u+.5,u),'yminus':(u,u-.5),'yplus':(u,u+.5),
           'inner_x_vertex_y':(u,v),'vertex_x_inner_y':(v,u)}
    out={}
    original=np.load(R/f'src/c70_centroid_lookup{diag}.npz')
    nominal=(original['ids'],original['weights'])
    for mode,(a,b) in specs.items():
        if mode=='centroid':
            pick,wt=nominal
        else:
            xx,yy=np.meshgrid(a,-b)
            coords=np.stack([xx.ravel(), yy.ravel()],1);si=tri.find_simplex(coords)
            t=tri.transform[si];w=np.einsum('nij,nj->ni',t[:,:2],coords-t[:,2])
            pick=tri.simplices[si].copy();wt=np.column_stack([w,1-w.sum(1)])
            outside=si<0;pick[outside]=nominal[0][outside];wt[outside]=nominal[1][outside]
        out[mode]=(torch.from_numpy(np.array(pick,dtype=np.int64)).cuda(),torch.from_numpy(np.array(wt,dtype=np.float32)).cuda())
    pi,wi=out['centroid']
    if diag==0:
        ws=wi.square();out['sharp']=(pi,ws/ws.sum(1,keepdim=True).clamp_min(1e-9))
        out['triangle_nearest']=(pi,F.one_hot(wi.argmax(1),3).float())
        # Cocircular centroid sites can have different interpolating triplets after order/precision changes.
        reverse=Delaunay(centers[::-1].astype('float64'),qhull_options='Qbb Qc Qz Q12')
        xx,yy=np.meshgrid(u,-u);coords=np.stack([xx.ravel(), yy.ravel()],1)
        si=reverse.find_simplex(coords);t=reverse.transform[si];w=np.einsum('nij,nj->ni',t[:,:2],coords-t[:,2])
        pick=(len(centers)-1-reverse.simplices[si]);wt=np.column_stack([w,1-w.sum(1)])
        bad=si<0;pick[bad]=nominal[0][bad];wt[bad]=nominal[1][bad]
        out['consistent_ties_flip']=(torch.from_numpy(pick.astype('int64')).cuda(),torch.from_numpy(wt.astype('float32')).cuda())
    return out

def change_height(h,k,xx,yy):
    if k in ('native','anti_native'):return h
    if k=='floor1':return h.floor()
    if k in ('round1','round025'):
        step=1. if k=='round1' else .25
        return (h/step).round()*step
    if k in ('u8','u10','u12'):
        step=h.max().clamp_min(.001)/(2**int(k[1:])-1)
        return (h/step).round()*step
    smooth=F.avg_pool2d(F.pad(h[None,None],(1,1,1,1),mode='replicate'),3,1)[0,0]
    if k=='smooth':return .5*(h+smooth)
    if k=='sharpen':return h+.75*(h-smooth)
    if k=='coarse2':
        low=F.interpolate(h[None,None],size=(128,128),mode='bilinear',align_corners=True)
        return F.interpolate(low,size=(256,256),mode='bilinear',align_corners=True)[0,0]
    sign=-1 if k=='unwarp' else 1
    if k=='stagger':return warp(h,xx+((yy.remainder(2))-.5)*.5,yy)
    return warp(h, xx+sign*(.333333-xx*(2/3)/255),yy+sign*(.333333-yy*(2/3)/255))

class Source(RegisteredTransport):
    def source_height(self,h,r,c):
        return self.tx_z

def native(engine, h, r, c, z):
    engine.tx_z=z
    direct, spec, diffuse, ht, x, y, zz = engine.raw(h,r,c)
    geom,_=engine.scene(ht)
    face=engine.faces; pos=engine.xy[face].mean(1)
    alt=ht.flatten()[face].mean(1)+1.
    dx=pos[:,0]-x*engine.spacing;dy=pos[:,1]-y*engine.spacing;dz=alt-zz
    dist=(dx.square()+dy.square()+dz.square()).clamp_min(.0625)
    gx=geom[:,1]/engine.spacing;gy=geom[:,2]/engine.spacing
    cs=((dz-gx*dx-gy*dy)/torch.sqrt(dist*(1+gx.square()+gy.square()))).clamp(-1,1)
    p=direct[:,:2]*dist[:,None];ex=(spec[:,:2]+diffuse[:,:2])*dist[:,None]
    nn=4e9/(4*math.pi)*geom[:,3]*cs.abs()/dist
    # Finite unit-area count and weakest face signal remain separate from averaged powers.
    return p.contiguous(),ex.contiguous(),direct[:,2].clamp(0,1),cs,nn,torch.stack([gx,gy],1)

def observe(data, lookup, pix):
    ix,w=lookup;idx=ix[pix];weights=w[pix];ww=weights[...,None]
    p,ex,vis,cs,nn,grad=data
    direct=p[idx];extra=ex[idx];s=(direct+extra).clamp_min(1e-14);v=vis[idx];c=cs[idx]
    avg=lambda v:(v*ww).sum(1)
    db=lambda x:-10*torch.log10(x.clamp_min(1e-14))
    risk=(1/(1+nn[idx]/6))
    sq=lambda v:(v*weights).sum(1)
    weak=(db(extra+.008*direct)-db(s)).clamp(-35,65)
    vals=torch.cat([db(avg(direct)).clamp(-12,100),db(avg(s)).clamp(-12,100),
        avg(db(s)).clamp(-12,100),sq(v)[:,None],sq(risk)[:,None],sq(1-v)[:,None],
        sq(torch.log10(c.abs().clamp_min(1e-6)))[:,None],sq((c.abs()<.015).float())[:,None],
        avg(weak*risk[:,:,None]),avg(grad[idx]).clamp(-15,15),weights.max(1).values[:,None],
        torch.where(weights>.05,c.abs(),1).amin(1)[:,None],(4*sq((c<0).float())*(1-sq((c<0).float())))[:,None]],1)
    assert vals.shape[1]==len(COL)
    return vals

def main():
    import csv
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=3)
    ap.add_argument('--stage',choices=['screen','confirm'],default='screen');args=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(3)
    records=json.load(open(A/'protocol.json'))['panel']
    rec=[e for e in records if (e['group']=='confirm')==(args.stage=='confirm')][args.shard::args.of]
    panel=np.load(R/'audit/c78_clear/train_panel.npz')
    rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));heights=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
    if args.stage=='screen':
        names=[('native',q) for q in QUERY]+[(v,'centroid') for v in VERTEX if v!='native']
    else:names=[tuple(v) for v in json.load(open(A/'selected.json'))['keep']]
    dest=A/(args.stage+str(args.shard));dest.mkdir(exist_ok=True)
    engine={diag:Source(physical=1,anti=diag,smooth=1,direct=3,spec=3,diffuse=96) for diag in set(int(v=='anti_native') for v,q in names)}
    lookup={d:queries(d) for d in engine}
    yy,xx=torch.meshgrid(torch.arange(256,device='cuda').float(),torch.arange(256,device='cuda').float(),indexing='ij')
    arr=np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype=np.float16,shape=(len(rec),len(names),2048,len(COL)))
    t=time.monotonic()
    for j,e in enumerate(rec):
        row=rows[e['index']];r,c=int(row['tx_row']),int(row['tx_col'])
        h=heights[int(row['hm_row'])];ht=torch.from_numpy(np.array(h)).cuda();z=float(np.floor(h[min(r+1,255),c])+3)
        px=torch.from_numpy(panel['pixels'][e['pos']].astype(np.int64)).cuda()
        ret=torch.empty((len(names),len(px),len(COL)),device='cuda')
        for v in dict.fromkeys(k for k,q in names):
            diag=int(v=='anti_native')
            surface=change_height(ht,v,xx,yy)
            data=native(engine[diag],surface.cpu().numpy(),r,c,z)
            for i,(key,q) in enumerate(names):
                if key==v:ret[i]=observe(data,lookup[diag][q],px)
        assert torch.isfinite(ret).all()
        arr[j]=ret.half().cpu().numpy()
        if j%16==0:print('done',j+1,'of',len(rec),'seconds',round(time.monotonic()-t,1),flush=True)
    arr.flush()
    atomic_json(dest/'complete.json',dict(gpu=gpu,records=rec,names=names,columns=COL,seconds=time.monotonic()-t))

if __name__=='__main__':main()
