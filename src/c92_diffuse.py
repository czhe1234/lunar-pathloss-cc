"""Multiple diffuse encounters over an entire native triangulated terrain."""
import argparse,csv,ctypes,json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c67_physics import quadrature,directions
from src.c69_registered import RegisteredTransport
A=R/'audit/c92_diffuse'
class Field:
    def __init__(self):
        self.rt=RegisteredTransport(physical=1,anti=0,smooth=1,direct=1,spec=1,diffuse=1);self.q=quadrature(1)
        self.dirs={n:directions(n) for n in (96,384)}
        lib=ctypes.CDLL(str(R/'src/libc92_diffuse.so'));self.lib=lib
        p=ctypes.c_void_p;i=ctypes.c_int;s=ctypes.c_float
        self.ini=lib.initial92;self.ini.argtypes=[p,p,p,p,s,s,s,s,p,p,p,p];self.ini.restype=i
        self.move=lib.exchange92;self.move.argtypes=[p,p,p,p,i,i,p,i,i,p,p,p,i,s,p,p];self.move.restype=i
    def seed(self,h,r,c):
        self.h=torch.as_tensor(np.array(h),device='cuda');self.g,self.n=self.rt.scene(self.h);self.top=float(h.max())+2
        self.tx=(c*255/256,(r+1)*255/256,self.rt.source_height(h,r,c))
        d=torch.empty_like(self.g);em=torch.empty_like(d);out=torch.empty_like(d)
        err=self.ini(self.g.data_ptr(),self.n.data_ptr(),self.rt.material.data_ptr(),self.q.data_ptr(),*self.tx,self.top,
                d.data_ptr(),em.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        return out
    def proposal(self,f):
        area=self.g[:,3];p=(f[:,:2]+f[:,2:])*area[:,None];total=p.sum(0)
        prob=.46*(p/total.clamp_min(1e-27)).sum(1)+.08*area/area.sum()
        prob=(prob/prob.sum()).contiguous();cdf=prob.double().cumsum(0).float().contiguous();cdf[-1]=1
        return prob,cdf
    def propagate(self,f,proposal,ids,ground,seed):
        prob,cdf=proposal;n=len(self.g) if ground else len(ids);nd,nl=(96,48) if ground else (384,128)
        out=torch.empty((n,4),device='cuda')
        err=self.move(self.g.data_ptr(),self.n.data_ptr(),self.rt.material.data_ptr(),ids.data_ptr(),n,int(ground),
           self.dirs[nd].data_ptr(),nd,nl,f.data_ptr(),prob.data_ptr(),cdf.data_ptr(),seed,self.top,
           out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        return out
    def features(self,h,r,c,px,seed,check=False):
        field=self.seed(h,r,c);p=torch.as_tensor(px.astype('int64'),device='cuda');ix=self.rt.ids[p];wt=self.rt.weights[p]
        uf,inv=torch.unique(ix,return_inverse=True);ids=uf.int()
        outputs=[];energy=[]
        for k in range(3):
            prop=self.proposal(field)
            rx=self.propagate(field,prop,ids,False,seed+41*k)
            v=(rx[inv,:2]*wt[:,:,None]).sum(1)
            outputs.append(v);energy.append(float((field.sum(1)*self.g[:,3]).sum()))
            if k<2:field=self.propagate(field,prop,ids,True,seed+43*k+7)
        v=torch.cat(outputs,1);assert torch.isfinite(v).all() and (v>=0).all()
        return v.cpu().numpy(),energy
    def check(self):
        pix=np.array([100*256+94,135*256+128,115*256+135,100*256+103,150*256+152])
        flat=np.zeros((256,256),'float32');v,e=self.features(flat,110,115,pix,210)
        assert (v[:,:2]>0).all() and np.max(v[:,2:])<1e-10,(v,e)
        y,x=np.indices(flat.shape);h=(.35*np.abs(x-126)+.04*np.abs(y-128)).astype('float32')
        v2,e2=self.features(h,120,103,pix,210)
        assert (v2[:,2:4]>0).any() and 0<e2[2]<e2[1]<e2[0],e2
        return dict(flat_second_max=float(np.max(v[:,2:])),concave_integrated_ground_flux=e2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--of',type=int,default=3)
    ap.add_argument('--stage',choices=['screen','confirm'],default='screen');ap.add_argument('--check',action='store_true');a=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);model=Field();qa=model.check()
    src=R/'audit/c91_image'/(a.stage+str(a.shard));meta=json.load(open(src/'done.json'));records=meta['rows'];loc=np.load(src/'loc.npy')
    pixels=np.load(R/'audit/c78_clear/train_panel.npz')['pixels'];hh=np.load(DATA/'train_hm.npy',mmap_mode='r');rows=list(csv.DictReader(open(DATA/'train_index.csv')))
    out=A/(a.stage+str(a.shard));out.mkdir(exist_ok=True);ts=time.monotonic()
    subset=records[:1] if a.check else records
    if not a.check:arr=np.lib.format.open_memmap(out/'power.npy',mode='w+',shape=(len(records),512,6),dtype='float32')
    for j,e in enumerate(subset):
        row=rows[e['index']];px=pixels[e['pos'],loc[j]]
        v,energy=model.features(hh[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']),px,920014+e['index'])
        if a.check:print('qa',qa,'one terrain seconds',time.monotonic()-ts,'ground flux',energy,flush=True)
        else:arr[j]=v
        if j%14==0:print(j+1,len(subset),round(time.monotonic()-ts,1),flush=True)
    if not a.check:
        arr.flush();atomic_json(out/'done.json',dict(gpu=gpu,checks=qa,positions=[e['pos'] for e in records],elapsed=time.monotonic()-ts))
if __name__=='__main__':main()
