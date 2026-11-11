"""Estimate finite-ray hit risk from triangle shape, orientation and lattice phase."""
import argparse, ctypes, csv, json, time, sys
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
A=R/'audit/c79_receiver'
SIZES=[1_000_000,4_000_000,40_000_000,400_000_000,1_000_000_000,4_000_000_000]

def normals(n):
    # Reduce two real vectors of the exact dual lattice. Short combinations also resolve long, narrow projected faces.
    phi=(1+np.sqrt(np.longdouble(5)))/2
    a=np.array([1.,-(n-1)/phi],dtype=np.longdouble);b=np.array([0.,n-1],dtype=np.longdouble)
    for _ in range(60):
        if b@b<a@a:a,b=b,a
        m=int(np.rint((a@b)/(a@a)))
        if m==0:break
        b=b-m*a
    v=[]
    for u in range(0,8):
        for k in range(-7,8):
            if (u==0 and k!=1) or np.gcd(abs(u),abs(k))!=1:continue
            v.append(u*a+k*b)
    return np.asarray(v,np.float64)

class Aperture:
    def __init__(self):
        self.n=torch.tensor(SIZES,device='cuda',dtype=torch.float64)
        self.dual=torch.from_numpy(np.stack([normals(n) for n in SIZES])).cuda()
        ds=np.load(R/'src/c70_centroid_lookup0.npz')
        self.i=torch.from_numpy(ds['ids'].astype(np.int32)).cuda()
        self.w=torch.from_numpy(ds['weights'].astype(np.float32)).cuda()
        self.fn=ctypes.CDLL(str(R/'src/libc80_aperture.so')).aperture80
        p=ctypes.c_void_p;s=ctypes.c_float;i=ctypes.c_int
        self.fn.argtypes=[p,p,i,s,s,s,p,i,p,i,p,p];self.fn.restype=i
    def __call__(self,h,r,c,pix):
        t=torch.as_tensor(np.array(h,copy=True),device='cuda',dtype=torch.float32);px=torch.as_tensor(pix.astype('int64'),device='cuda')
        ids=self.i[px].contiguous();w=self.w[px];z=float(np.floor(h[min(r+1,255),c])+3)
        y=torch.empty((len(self.n),len(px),3,8),dtype=torch.float32,device='cuda')
        status=self.fn(t.data_ptr(),ids.data_ptr(),ids.numel(),c*255/256.,(r+1)*255/256.,z,self.dual.data_ptr(),self.dual.shape[1],self.n.data_ptr(),len(self.n),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if status:raise RuntimeError(status)
        avg=(y*w[None,:,:,None]).sum(2)
        # Logarithm before/after area mixing distinguishes native misses from raster-average power dips.
        signal=(1-y[...,3])*y[...,4]
        db=-10*torch.log10(signal.clamp_min(.003))
        extra=torch.stack([(db*w[None]).sum(2),-10*torch.log10((signal*w[None]).sum(2).clamp_min(.003)),
                 torch.where(w[None]>=.05,y[...,2],0).amax(2),torch.where(w[None]>=.05,y[...,6],0).amax(2)],2)
        out=torch.cat([avg,extra],2);assert torch.isfinite(out).all()
        return out.permute(1,0,2).contiguous()
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--of',type=int,default=3);args=p.parse_args()
    g=gpu_guard();engine=Aperture()
    plan=json.load(open(A/'protocol.json'))['panel'][args.shard::args.of]
    pan=np.load(R/'audit/c78_clear/train_panel.npz');rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));hs=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
    out=A/('angular'+str(args.shard));out.mkdir(exist_ok=True);t=time.monotonic()
    arr=np.lib.format.open_memmap(out/'x.npy',mode='w+',dtype=np.float16,shape=(len(plan),2048,len(SIZES),12))
    for j,rec in enumerate(plan):
        row=rows[rec['index']];h=hs[int(row['hm_row'])]
        arr[j]=engine(h,int(row['tx_row']),int(row['tx_col']),pan['pixels'][rec['pos']]).half().cpu().numpy()
        if j%80==0:print(j+1,len(plan),'elapsed',round(time.monotonic()-t,1),flush=True)
    arr.flush();atomic_json(out/'complete.json',dict(gpu=g,records=plan,sizes=SIZES,seconds=time.monotonic()-t))
if __name__=='__main__':main()
