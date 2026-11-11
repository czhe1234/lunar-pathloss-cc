"""Finite ray survival and the conditional mean of a sampled receiving cell."""
import argparse,sys,json,csv,time,ctypes
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c67_physics import quadrature
from src.c69_registered import RegisteredTransport
A=R/'audit/c89_survival'
class Survival:
    def __init__(self):
        self.rt=RegisteredTransport(physical=1,anti=0,smooth=1,direct=1,spec=1,diffuse=1)
        self.q=quadrature(3);self.lib=ctypes.CDLL(str(R/'src/libc89_branch.so'))
        i=ctypes.c_int;s=ctypes.c_float;p=ctypes.c_void_p
        self.fn=self.lib.sample89;self.fn.argtypes=[p,p,p,p,i,p,i,s,s,s,s,p,p];self.fn.restype=i
    def native(self,h,r,c,idx):
        v=torch.as_tensor(np.array(h),device='cuda');g,ns=self.rt.scene(v)
        z=self.rt.source_height(h,r,c);out=torch.empty((len(idx),17),device='cuda')
        err=self.fn(g.data_ptr(),ns.data_ptr(),self.rt.material.data_ptr(),idx.data_ptr(),len(idx),
                    self.q.data_ptr(),len(self.q),c*255/256,(r+1)*255/256,z,float(h.max())+2,
                    out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        return out
    def features(self,h,r,c,px):
        sel=torch.as_tensor(px.astype(np.int64),device='cuda');ii=self.rt.ids[sel];w=self.rt.weights[sel]
        uniq,inv=torch.unique(ii,return_inverse=True);v=self.native(h,r,c,uniq.int())[inv]
        def avg(x):return (x*w).sum(1)
        def loss(x):return (-10*torch.log10(x.clamp(1e-13,1e8))).clamp(-50,130)
        rays=v[:,:,0];cross=v[:,:,1];clear=v[:,:,2];ci=v[:,:,3];d=v[:,:,4]
        output=[avg(torch.log1p(rays)),avg(cross),avg(clear),avg(ci),avg(torch.log1p(d)),avg(1/(1+rays/5))]
        band_features=[[],[]]
        for b in (0,1):
            k=5+6*b
            power=v[:,:,k];lam=v[:,:,k+1];branch_rate=v[:,:,k+2];var=v[:,:,k+3]
            # Unobstructed direct paths do not resample their launch lattice when seeds change.
            open_part=(clear>.001)
            q=torch.where(open_part,torch.ones_like(lam),-torch.expm1(-lam.clamp(max=80)))
            q15=torch.where(open_part,torch.ones_like(lam),-torch.expm1(-(15*lam).clamp(max=80)))
            q60=torch.where(open_part,torch.ones_like(lam),-torch.expm1(-(60*lam).clamp(max=80)))
            # Interpolation across native areas is tested in linear power and dB;
            # zeros contribute to loss terms only via the model, never the released mask.
            mean=avg(power);cond=power/q.clamp_min(1e-12)
            plain=[loss(mean),avg(loss(power)),avg(torch.log1p(rays)),avg(clear),avg(loss(power))*avg(clear)]
            physical=[avg(torch.log1p(lam)),avg(q),avg(q15),avg(q60),loss(avg(cond))-loss(mean),
                      avg(loss(cond))-avg(loss(power)),avg(torch.log1p(var/(power.square()+1e-18)).clamp(0,25)),
                      avg(loss(branch_rate)),avg(1-q),torch.where(w>.05,q,torch.ones_like(q)).amin(1),
                      avg(torch.log1p(v[:,:,k+4])),avg(loss(v[:,:,k+5]))]
            output+=plain;band_features[b]=physical
        output+=band_features[0]+band_features[1]
        arr=torch.stack(output,1);assert arr.shape[1]==40 and torch.isfinite(arr).all()
        return arr
def check(engine):
    h=np.zeros((256,256),np.float32);px=np.array([257*50,257*110,257*225])
    x=engine.features(h,120,129,px).cpu().numpy()
    assert np.max(abs(x[:,2]-1))<1e-5 and np.max(abs(x[:,[6,11]]))<1e-4
    assert np.max(abs(x[:,17]-1))<1e-5 and np.max(abs(x[:,29]-1))<1e-5
    # Compare the analytic conditional estimator with independent Binomial samples.
    rng=np.random.default_rng(891130);n=14;prob=.035;tx=.023
    k=rng.binomial(n,prob,500000);obs=(k[k>0]/(n*prob)*tx).mean()
    theory=tx/(1-(1-prob)**n);assert abs(obs/theory-1)<.005
    return dict(flat_clear_and_bands='pass',conditional_binomial_relative_error=float(abs(obs/theory-1)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=3)
    a=ap.parse_args();gpu=gpu_guard();torch.set_num_threads(2);rt=Survival();qa=check(rt)
    source=np.load(R/'audit/c78_clear/train_panel.npz');groups=json.load(open(R/'audit/c88_mesh/base_index.json'))['groups']
    rows=list(csv.DictReader(open(DATA/'train_index.csv')));heights=np.load(DATA/'train_hm.npy',mmap_mode='r')
    positions=list(range(a.shard,len(source['ids']),a.of));out=A/f'shard{a.shard}';out.mkdir(parents=True,exist_ok=True)
    ar=np.lib.format.open_memmap(out/'features.npy',mode='w+',shape=(len(positions),2048,40),dtype='float16');ts=time.monotonic()
    for j,k in enumerate(positions):
        i=int(source['ids'][k]);r=rows[i]
        ar[j]=rt.features(heights[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']),source['pixels'][k]).half().cpu().numpy()
        if j%180==0:print('maps',j+1,'/',len(positions),'seconds',round(time.monotonic()-ts),flush=True)
    ar.flush();atomic_json(out/'done.json',dict(gpu=gpu,positions=positions,qa=qa,seconds=time.monotonic()-ts))
if __name__=='__main__':main()
