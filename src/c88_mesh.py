"""Input-only finite receiver mesh at its native resolution and per-cell diagonal."""
import argparse,csv,ctypes,json,sys,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c57_transmission import slab
from src.c70_geometry import GeometricMaps,qpoints
from scripts.c88_mesh_tables import ordered_faces
import torch
from torch.nn import functional as F
A=R/'audit/c88_mesh'
MODES=['256_fixed','256_opposite','256_checker','256_delaunay','256_roll','128_fixed','192_fixed','384_fixed','256_short','256_shortroll']
@lru_cache(maxsize=512)
def atlas(pattern):
    diag=((pattern>>np.arange(9))&1).reshape(3,3)
    faces=ordered_faces(diag)
    y,x=np.indices((4,4))
    center=np.stack([x.ravel()[faces].mean(1),y.ravel()[faces].mean(1)],1)
    return Delaunay(center)
def local_lookup(diag,pixels,nominal):
    m=diag.shape[0];p=np.asarray(pixels);y,x=np.unravel_index(p,(256,256))
    u=np.linspace(1/3,m-1/3,256);u,v=u[x],u[y];ix=u.astype(int);iy=v.astype(int)
    inside=(ix>0)&(iy>0)&(ix<m-1)&(iy<m-1)
    pat=np.zeros(len(p),np.int32)
    for k in range(9):
        pat|=diag[(iy+k//3-1).clip(0,m-1),(ix+k%3-1).clip(0,m-1)].astype(np.int32)<<k
    ids=nominal[0][p].copy();weights=nominal[1][p].copy()
    for b in np.unique(pat[inside]):
        idx=np.flatnonzero(inside&(pat==b));tri=atlas(int(b))
        pts=np.stack([u[idx]-ix[idx]+1,v[idx]-iy[idx]+1],1);s=tri.find_simplex(pts)
        assert (s>=0).all()
        t=tri.transform[s];w=np.einsum('nij,nj->ni',t[:,:2],pts-t[:,2]);f=tri.simplices[s]
        j=f%9;ids[idx]=(iy[idx,None]+j//3-1)*m+ix[idx,None]+j%3-1+(f//9)*m*m
        weights[idx]=np.column_stack([w,1-w.sum(1)])
    return ids,weights
class Mesh:
    def __init__(self):
        self.q=qpoints(3)
        c=torch.linspace(0.,1.,4097,device='cuda')
        f=torch.tensor([415e6,5.775e9],device='cuda');sig=torch.tensor([.0003562,.009324],device='cuda')
        ts,tp,_,_=slab(c,f,sig)
        self.slab=(-10*torch.log10(((ts.abs().square()+tp.abs().square())*.5).clamp(1e-11,1))).contiguous()
        self.lib=ctypes.CDLL(str(R/'src/libc88_geometry.so'))
        p=ctypes.c_void_p;i=ctypes.c_int;s=ctypes.c_float
        self.fn=self.lib.variable_geometry;self.fn.argtypes=[p,p,i,s,s,s,s,p,i,p,i,p,p,p];self.fn.restype=i
        self.tables={}
        for file in (A/'tables').glob('*.npz'):
            v=np.load(file);self.tables[file.stem]=(v['ids'],v['weights'],v['diag'])
    def native(self,h,diag,ids,x,y,z):
        m=len(diag);sp=256/m
        dg=torch.as_tensor(diag,device='cuda',dtype=torch.uint8).flatten().contiguous()
        flag=dg.bool()
        a=h[:-1,:-1].flatten();b=h[:-1,1:].flatten();c=h[1:,:-1].flatten();d=h[1:,1:].flatten()
        alt=torch.cat([a,torch.where(flag,b+c-d,a)])
        sx=torch.cat([b-a,d-c]);sy=torch.cat([torch.where(flag,c-a,d-b),torch.where(flag,d-b,c-a)])
        g=torch.stack([alt,sx,sy,.5*sp*sp*torch.sqrt(1+(sx/sp).square()+(sy/sp).square())],1).contiguous()
        face=torch.as_tensor(ids,device='cuda',dtype=torch.int32).contiguous()
        out=torch.empty((len(face),29),device='cuda')
        err=self.fn(g.data_ptr(),self.q.data_ptr(),len(self.q),x,y,z,sp,dg.data_ptr(),m,face.data_ptr(),len(face),
                    self.slab.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        return out
    def calc(self,height,r,c,mode,px):
        n=int(mode.split('_')[0]);m=n-1
        h=torch.from_numpy(np.array(height,dtype=np.float32)).cuda()
        z=float(np.floor(height[min(r+1,255),c])+3)
        if n!=256:h=F.interpolate(h[None,None],size=(n,n),mode='bilinear',align_corners=True)[0,0]
        if 'short' in mode:
            a=h[:-1,:-1];b=h[:-1,1:];o=h[1:,:-1];d=h[1:,1:]
            diag=(abs(b-o)<abs(d-a)).cpu().numpy().astype(np.uint8)
            if mode.endswith('roll'):diag=np.roll(diag,(37,59),(0,1)).copy()
            ii,w=local_lookup(diag,px,self.tables['256_fixed'])
        else:
            ii,w,diag=self.tables[mode];ii=ii[px];w=w[px]
        uf,inverse=np.unique(ii,return_inverse=True)
        nf=self.native(h,diag,uf,c*m/256,(r+1)*m/256,z)
        val=nf[torch.as_tensor(inverse.reshape(ii.shape),device='cuda')]
        wt=torch.as_tensor(w,device='cuda')[:,:,None]
        mean=(val*wt).sum(1)
        sel=[8,9,14,18,19,20,23,24,27,28]
        sd=((val[:,:,sel]-mean[:,None,sel]).square()*wt).sum(1).clamp_min(0).sqrt()
        # Area-averaged linear power and interpolation in dB must stay distinguishable.
        loss=-10*torch.log10(val[:,:,27:29].clamp_min(1e-11))
        db=-10*torch.log10(mean[:,27:29].clamp_min(1e-11))
        exph=torch.expm1((val[:,:,20:21]*10).clamp(0,15))
        miss=(wt/(1+exph/4)).sum(1)
        return torch.cat([mean[:,:27],sd,db,(loss*wt).sum(1),miss,wt.max(1).values],1)
    def check(self):
        yy,xx=np.indices((256,256));h=(.03*xx+.1*yy+6*np.sin(xx/11)*np.cos(yy/15)).astype('float32');r,c=130,100
        hh=torch.from_numpy(h).cuda();p=np.random.default_rng(332).choice(130050,2800,replace=False)
        old=GeometricMaps().native(hh,r,c)[0][p]
        dg=np.zeros((255,255),np.uint8);new=self.native(hh,dg,p,c*255/256,(r+1)*255/256,float(np.floor(h[r+1,c])+3))
        err=(new[:,:27]-old).abs().max().item()
        assert err<.001 and torch.isfinite(new).all(),err
        px=np.arange(65000).reshape(-1)[::41];ids,w=local_lookup(dg,px,self.tables['256_fixed'])
        cx=ids%65025%255+np.where(ids<65025,2/3,1/3);cy=ids%65025//255+np.where(ids<65025,1/3,2/3)
        iy,ix=np.unravel_index(px,(256,256));good=(ix>2)&(iy>2)&(ix<253)&(iy<253);u=np.linspace(1/3,255-1/3,256)
        errxy=max(np.max(abs((cx*w).sum(1)[good]-u[ix][good])),np.max(abs((cy*w).sum(1)[good]-u[iy][good])))
        assert errxy<.00003,errxy
        return dict(nominal_max_difference=float(err),interior_barycentric_xy_max_error=float(errxy))
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--of',type=int,default=3)
    p.add_argument('--stage',choices=['screen','confirm'],default='screen');a=p.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);engine=Mesh()
    assert len(engine.tables)==8
    checks=engine.check()
    records=json.load(open(R/'audit/c79_receiver/protocol.json'))['panel']
    records=[e for e in records if (e['group']=='confirm')==(a.stage=='confirm')][a.shard::a.of]
    names=MODES if a.stage=='screen' else json.load(open(A/'selection.json'))['physical_modes']
    px=np.load(R/'audit/c78_clear/train_panel.npz')['pixels'];rows=list(csv.DictReader(open(DATA/'train_index.csv')))
    heights=np.load(DATA/'train_hm.npy',mmap_mode='r')
    out=A/f'{a.stage}_{a.shard}';out.mkdir(exist_ok=True);t=time.monotonic()
    for j,e in enumerate(records):
        i=e['index'];r=rows[i];pix=px[e['pos']].astype(np.int64)
        feats=torch.stack([engine.calc(heights[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']),name,pix) for name in names])
        assert torch.isfinite(feats).all()
        ar=feats.half().cpu().numpy()
        if j==0:arr=np.lib.format.open_memmap(out/'features.npy','w+',dtype='float16',shape=(len(records),*ar.shape))
        arr[j]=ar
        if j%10==0:print('maps',j+1,'/',len(records),'seconds',round(time.monotonic()-t),flush=True)
    arr.flush();atomic_json(out/'done.json',dict(records=records,modes=names,checks=checks,gpu=gpu,elapsed=time.monotonic()-t))
if __name__=='__main__':main()
