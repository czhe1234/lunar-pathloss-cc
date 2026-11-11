"""Isolate direct, specular and first diffuse components on the same inferred receiver triangles."""
import argparse,ctypes,csv,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c69_registered import RegisteredTransport
from src.c67_physics import quadrature,directions
import numpy as np
import torch
A=R/'audit/c78_clear'
TERMS=['direct4','direct16','spec16','spec64','spec_flat64','diff128','diff512','diff_mis','diff_light','native_db_old','native_db_mis']
COLUMNS=[term+'_'+b for term in TERMS for b in ('415','58')]
class Components:
    def __init__(self):
        self.engine=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=128)
        self.lib=ctypes.CDLL(str(R/'src/libc78_power.so'));self.fn=self.lib.refinements78
        P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_float
        self.fn.argtypes=[P,P,P,P,I,P,I,P,I,P,I,I,P,P,P,I,S,S,S,S,S,I,I,P,P,P,P,P,P];self.fn.restype=I
        self.primary=quadrature(4);self.spec=[quadrature(8),quadrature(12)]
        self.dirs=[directions(512),directions(1536)]
    @torch.no_grad()
    def compute(self,h,r,c,pixels,seed,refine=False,extra=False):
        eng=self.engine;d0,sp0,df0,ht,x,y,z,em=eng.raw(h,r,c,return_emission=True)
        g,norm=eng.scene(ht);area=g[:,3];flux=(em[:,:2]*area[:,None]).clamp_min(0)
        total=flux.sum(0)
        p=.49*(flux[:,0]/total[0].clamp_min(1e-28)+flux[:,1]/total[1].clamp_min(1e-28))+.02*area/area.sum()
        p=(p/p.sum()).contiguous();cdf=p.double().cumsum(0).float().contiguous();cdf[-1]=1
        pix=torch.as_tensor(pixels.astype(np.int64),device='cuda')
        idx=eng.ids[pix];w=eng.weights[pix];target=idx.int().flatten().contiguous()
        d=torch.empty_like(d0);tmp=torch.empty_like(d0);s1=torch.zeros_like(sp0);sf=torch.zeros_like(sp0);df=torch.empty((len(target),6),device='cuda')
        spec=self.spec[int(refine)];dirs=self.dirs[int(refine)];nl=384 if refine else 128
        code=self.fn(g.data_ptr(),norm.data_ptr(),eng.material.data_ptr(),self.primary.data_ptr(),len(self.primary),spec.data_ptr(),len(spec),
                     target.data_ptr(),len(target),dirs.data_ptr(),len(dirs),nl,em.data_ptr(),p.data_ptr(),cdf.data_ptr(),seed,
                     x,y,z,eng.spacing,float(h.max()),0,1,d.data_ptr(),tmp.data_ptr(),s1.data_ptr(),sf.data_ptr(),df.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(code)
        ww=w[:,:,None]
        read=lambda t:(t[idx,:2]*ww).sum(1)
        diffout=df.reshape(-1,3,6);di=(diffout*ww).sum(1)
        old=d0[idx,:2]+sp0[idx,:2]+df0[idx,:2]
        new=d[idx,:2]+s1[idx,:2]+diffout[:,:,2:4]
        log=lambda val:-10*torch.log10(val.clamp_min(1e-30))
        out=torch.cat((read(d0),read(d),read(sp0),read(s1),read(sf),read(df0),di,(log(old)*ww).sum(1),(log(new)*ww).sum(1)),1)
        assert torch.isfinite(out).all()
        if extra:return out.cpu().numpy(),(em,g,idx,w,s1,sf)
        return out.cpu().numpy()
def check(eng):
    e=eng.engine;hh=np.zeros((256,256),np.float32);r=c=128
    yy,xx=np.mgrid[:256,:256];usable=np.flatnonzero(((xx-128)**2+(yy-128)**2>10**2)&((xx-128)**2+(yy-128)**2<75**2))
    pixels=np.random.default_rng(4197).choice(usable,64,replace=False)
    out,(em,g,idx,w,smooth,flat)=eng.compute(hh,r,c,pixels,117,refine=True,extra=True)
    centers=e.xy[e.faces].mean(1)
    exact=[]
    for tri in idx:
        # Direct area quadrature, independent of the directional and source sampling schemes.
        r2=(centers[tri,None,:]-centers[None,:,:]).square().sum(2)+1
        exact.append(((em[None,:,:2]*g[None,:,3:4]/(torch.pi*r2[:,:,None]**1.5)).sum(1)))
    reference=(torch.stack(exact)*w[:,:,None]).sum(1).cpu().numpy()
    ratio=(out[:,14:16]/reference).mean(0)
    assert np.max(np.abs(ratio-1))<.10 and np.max(np.abs(out[:,6:8]-out[:,8:10]))<1e-5
    print('flat diffuse / independent area integral',ratio.tolist(),flush=True)
    return ratio.tolist()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='train',choices=['train','val']);p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=3);p.add_argument('--check',action='store_true');a=p.parse_args()
    gpu=gpu_guard();eng=Components()
    if a.check:
        ratio=check(eng);atomic_json(A/f"power_check_{gpu['uuid']}.json",dict(ratio=ratio,gpu=gpu));return
    panel=np.load(A/f'{a.split}_panel.npz');ids=panel['ids'];pp=panel['pixels'];pos=np.arange(len(ids))[a.shard::a.of]
    rng=np.random.default_rng(787142)
    _,first=np.unique(panel['terrain'],return_index=True);sel=set(rng.choice(np.sort(first),48,replace=False).tolist())
    rows=list(csv.DictReader((DATA/f'{a.split}_index.csv').open()));heights=np.load(DATA/f'{a.split}_hm.npy',mmap_mode='r')
    out=A/f'{a.split}_power_{a.shard}of{a.of}';out.mkdir(exist_ok=False);np.save(out/'positions.npy',pos)
    m=np.lib.format.open_memmap(out/'power.npy','w+',dtype='float32',shape=(len(pos),pp.shape[1],len(COLUMNS)))
    sens=[];si=[];t=time.monotonic()
    for k,j in enumerate(pos):
        row=rows[ids[j]];h=heights[int(row['hm_row'])];r,c=int(row['tx_row']),int(row['tx_col'])
        m[k]=eng.compute(h,r,c,pp[j],int(ids[j])+71019)
        if j in sel:
            sens.append(eng.compute(h,r,c,pp[j],int(ids[j])+556319,refine=True));si.append(j)
        if k%40==0 or k+1==len(pos):
            v=dict(maps=k+1,total=len(pos),seconds=round(time.monotonic()-t,1));atomic_json(out/'progress.json',v);print(v,flush=True)
    m.flush();np.savez(out/'sensitivity.npz',positions=si,power=np.stack(sens));atomic_json(out/'complete.json',dict(gpu=gpu,columns=COLUMNS,source='H and TX only; positive components in inverse square metres, dB columns exclude frequency constant',maps=len(pos)))
if __name__=='__main__':main()
