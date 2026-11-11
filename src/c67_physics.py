"""Independently computed one-bounce transport and receiver-triangle interpolation."""
import argparse
import csv
import ctypes
import json
import time
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,atomic_json,DATA
import numpy as np
import torch
from torch.nn.functional import normalize
from scipy.spatial import cKDTree
from src.c66_lattice import Mesh
from src.c57_transmission import slab

def quadrature(n):
    out=[]
    for i in range(n):
        for j in range(n-i):
            for offset in ((1/3,2/3) if i+j+1<n else (1/3,)):
                u=(i+offset)/n;v=(j+offset)/n;out.append([1-u-v,u,v])
    assert len(out)==n*n
    return torch.tensor(out,dtype=torch.float32,device='cuda')
def directions(n):
    i=np.arange(n);z=1-2*(i+.5)/n;t=2*np.pi*i*(np.sqrt(5)-1)/2
    r=np.sqrt(1-z*z)
    return torch.tensor(np.column_stack((r*np.cos(t),r*np.sin(t),z)),dtype=torch.float32,device='cuda')

class Transport:
    def __init__(self,physical=1,anti=0,smooth=1,direct=1,spec=2,diffuse=48):
        self.anti=anti; self.smooth=smooth;self.physical=physical
        self.spacing=256/255 if physical else 1.
        mesh=Mesh(physical,anti)
        self.faces=torch.tensor(mesh.faces,dtype=torch.long,device='cuda')
        axis=np.linspace(1/3,255-1/3,256) if physical else np.arange(256)
        qx,qy=np.meshgrid(axis,axis)
        self.qx=torch.tensor(qx.reshape(-1),dtype=torch.float32,device='cuda')
        self.qy=torch.tensor(qy.reshape(-1),dtype=torch.float32,device='cuda')
        xy=np.stack(np.meshgrid(np.arange(256),np.arange(256)),axis=-1).reshape(-1,2)
        self.xy=torch.tensor(xy,dtype=torch.float32,device='cuda')*self.spacing
        ok=mesh.weights.sum(1)>.5
        if (~ok).any():
            _,near=cKDTree(xy[mesh.faces].mean(1)).query(np.column_stack((qx[~ok.reshape(256,256)],qy[~ok.reshape(256,256)])))
            mesh.neighbors[~ok]=near[:,None];mesh.weights[~ok]=[1,0,0]
        self.ids=torch.tensor(mesh.neighbors,device='cuda',dtype=torch.long)
        self.weights=torch.tensor(mesh.weights,device='cuda',dtype=torch.float32)
        self.support=torch.tensor(ok,device='cuda',dtype=torch.float32)
        self.primary=quadrature(direct);self.spec=quadrature(spec);self.dirs=directions(diffuse)
        c=torch.linspace(0.,1.,4097,device='cuda')
        f=torch.tensor([415e6,5.775e9],device='cuda');sig=torch.tensor([.0003562,.009324],device='cuda')
        ts,tp,rs,rp=slab(c,f,sig)
        self.material=torch.stack((ts.real,ts.imag,tp.real,tp.imag,rs.abs().square(),rp.abs().square()),dim=-1).permute(1,0,2).contiguous()
        self.lib=ctypes.CDLL(str(ROOT/'src/libc67_transport.so'))
        P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_float
        self.lib.transport.argtypes=[P,P,P,P,I,P,I,P,I,S,S,S,S,S,I,I,P,P,P,P,P]
        self.lib.transport.restype=I

    def scene(self,h):
        h=h.reshape(256,256)
        a=h[:-1,:-1].reshape(-1);b=h[:-1,1:].reshape(-1);c=h[1:,:-1].reshape(-1);d=h[1:,1:].reshape(-1)
        if self.anti:
            z=torch.cat((a,b+c-d));sx=torch.cat((b-a,d-c));sy=torch.cat((c-a,d-b))
        else:
            z=torch.cat((a,a));sx=torch.cat((b-a,d-c));sy=torch.cat((d-b,c-a))
        normal=torch.stack((-sx/self.spacing,-sy/self.spacing,torch.ones_like(z)),1)
        length=normal.norm(dim=1);g=torch.stack((z,sx,sy,length*(.5*self.spacing**2)),dim=1).contiguous()
        unit=normal/length[:,None];vn=torch.zeros((65536,3),device='cuda')
        if self.smooth:
            v=torch.cat((self.xy,h.reshape(-1,1)),1)[self.faces]
            for i in range(3):
                u=v[:,(i+1)%3]-v[:,i];w=v[:,(i+2)%3]-v[:,i]
                theta=torch.atan2(torch.linalg.cross(u,w).norm(dim=1),(u*w).sum(dim=1))
                vn.index_add_(0,self.faces[:,i],unit*theta[:,None])
            vn=normalize(vn,dim=1)
        return g,vn
    def source_height(self,h,row,col):
        return float(np.floor(h[row,col])+3)
    @torch.no_grad()
    def raw(self,h,row,col,return_emission=False):
        top=float(np.max(h));z=self.source_height(h,row,col)
        h=torch.as_tensor(np.array(h,copy=True),device='cuda',dtype=torch.float32)
        x=col*255/256 if self.physical else col;y=(row+1)*255/256 if self.physical else row
        g,vn=self.scene(h)
        size=len(self.faces);d=torch.empty((size,4),device='cuda');e=torch.empty_like(d);s=torch.zeros((size,3),device='cuda');a=torch.empty_like(d)
        ptr=lambda t:t.data_ptr()
        err=self.lib.transport(ptr(g),ptr(vn),ptr(self.material),ptr(self.primary),len(self.primary),
            ptr(self.spec),len(self.spec),ptr(self.dirs),len(self.dirs),x,y,z,self.spacing,top,
            self.anti,self.smooth,ptr(d),ptr(e),ptr(s),ptr(a),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(f'transport launch failed: {err}')
        result=(d,s,a,h,x,y,z)
        return (*result,e) if return_emission else result
    def interpolate(self,v):
        return (v[self.ids]*self.weights[:,:,None]).sum(1)
    @torch.no_grad()
    def features(self,h,row,col):
        direct,spec,diff,h,x,y,z=self.raw(h,row,col)
        qp=torch.stack((self.qx/127.5-1,self.qy/127.5-1),1).reshape(1,256,256,2)
        hq=torch.nn.functional.grid_sample(h[None,None],qp,align_corners=True).reshape(-1)
        r2=((self.qx-x).square()+(self.qy-y).square())*self.spacing**2+(hq+1-z).square()
        r2=r2.clamp_min(.0625)[:,None]
        power=torch.cat((direct[:,:2],spec[:,:2],diff[:,:2]),1)
        total=direct[:,:2]+spec[:,:2]+diff[:,:2]
        log=lambda v:torch.log10(v.clamp(1e-12,1e8))
        effects=log(self.interpolate(torch.cat((power,total),1))*r2)/4
        logmean=(self.interpolate(log(total))+log(r2))/4
        diagnostic=self.interpolate(torch.stack((direct[:,2],torch.log1p(direct[:,3])/3,torch.log1p(spec[:,2]),
            diff[:,0].square()/diff[:,2].clamp_min(1e-30)/len(self.dirs),
            diff[:,1].square()/diff[:,3].clamp_min(1e-30)/len(self.dirs)),1))
        out=torch.cat((effects,logmean,diagnostic,self.support[:,None]),1)
        return out.T.reshape(-1,256,256).contiguous()
def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--dir',default='features/c67_transport')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1)
    p.add_argument('--direct',type=int,default=1);p.add_argument('--spec',type=int,default=2);p.add_argument('--diffuse',type=int,default=48)
    p.add_argument('--anti',type=int,default=0);p.add_argument('--smooth',type=int,default=1)
    p.add_argument('--panel',default='');args=p.parse_args();gpu=gpu_guard()
    engine=Transport(anti=args.anti,smooth=args.smooth,direct=args.direct,spec=args.spec,diffuse=args.diffuse)
    rows=list(csv.DictReader((DATA/'train_index.csv').open()));heights=np.load(DATA/'train_hm.npy',mmap_mode='r')
    panel=json.loads((ROOT/'audit/c66_transport/panels.json').read_text())
    if args.check:
        flat=np.zeros((256,256),np.float32)
        start=time.monotonic();direct,spec,diff,h,x,y,z=engine.raw(flat,128,128)
        ctr=(engine.xy[engine.faces]).mean(1);dist=(ctr[:,0]-x*engine.spacing).square()+(ctr[:,1]-y*engine.spacing).square()
        shell=(dist>=10**2)&(dist<=80**2)
        cos=4/(dist+16).sqrt()
        lut=engine.material[:,(cos*4096).long().clamp_max(4096),5].T
        mirror=(1-.632**2)*lut/(dist[:,None]+16)
        stats={'flat_direct_median':(direct[shell,:2]*(dist[shell,None]+4)).median(0).values.tolist(),
               'flat_los_min':float(direct[:,2].min()),'flat_spec_over_analytic_annulus':(spec[shell,:2].sum(0)/mirror[shell].sum(0)).tolist(),
               'flat_diffuse_positive':(diff[shell,:2]>0).float().mean(0).tolist()}
        ridge=np.broadcast_to(np.maximum(0,12-np.abs(np.arange(256)-128)),(256,256)).astype(np.float32)
        ds,*_=engine.raw(ridge,100,100)
        plane=(ctr[:,0]/engine.spacing>146)&(ctr[:,0]/engine.spacing<166)&(ctr[:,1]/engine.spacing>100)&(ctr[:,1]/engine.spacing<102)
        stats['ridge_crossings_mean']=ds[plane,3].mean().item()
        stats['ridge_ratio_high_over_low']=float(ds[plane,1].sum()/ds[plane,0].sum())
        assert stats['flat_los_min']==1 and max(abs(v-1) for v in stats['flat_direct_median'])<.01
        assert max(abs(v-1) for v in stats['flat_spec_over_analytic_annulus'])<.12
        assert stats['ridge_crossings_mean']>=1.9 and stats['ridge_ratio_high_over_low']<.1
        for i in panel['fit_first'][:3]:
            t=time.monotonic();r=rows[i];out=engine.features(heights[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']))
            assert torch.isfinite(out).all()
            stats.setdefault('map_seconds',[]).append(time.monotonic()-t)
        stats['hardware']=gpu
        atomic_json(ROOT/'audit/c66_transport/c67_renderer_check.json',stats);print(json.dumps(stats),flush=True)
        return
    ids=json.loads((ROOT/args.panel).read_text()) if args.panel else panel['dev_maps']+panel['fit_maps']
    ix=ids[args.shard::args.of];out=ROOT/args.dir/f'part{args.shard}of{args.of}'
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'indices.json',ix)
    arr=np.lib.format.open_memmap(out/'features.npy',mode='w+',shape=(len(ix),16,256,256),dtype=np.float16)
    t=time.monotonic()
    for j,i in enumerate(ix):
        r=rows[i];x=engine.features(heights[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']))
        val=x.half().cpu().numpy()
        if not np.isfinite(val).all():raise RuntimeError(f'nonfinite features at {i}')
        arr[j]=val
        if j%32==0 or j==len(ix)-1:
            progress={'maps':j+1,'total':len(ix),'seconds':round(time.monotonic()-t,1)}
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    arr.flush();atomic_json(out/'complete.json',dict(vars(args),count=len(ix),gpu=gpu))
if __name__=='__main__':main()
