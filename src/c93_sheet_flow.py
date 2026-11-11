"""Trace incident and repeated receiver-sheet flux; file contains only H/TX-derived quantities."""
import sys,json,time,argparse,ctypes,csv
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c78_geometry import planes
from src.c70_geometry import qpoints
A=R/'audit/c93_joint_miss'
class Flow:
    def __init__(self):
        self.lib=ctypes.CDLL(str(R/'src/libc93_sheet_flow.so'));self.fn=self.lib.receiver_flow93
        p=ctypes.c_void_p;s=ctypes.c_float;i=ctypes.c_int
        self.fn.argtypes=[p,p,i,s,s,s,s,p,p];self.fn.restype=i
        lookup=np.load(R/'src/c70_centroid_lookup0.npz')
        self.idx=torch.as_tensor(lookup['ids'].astype('int64'),device='cuda');self.w=torch.as_tensor(lookup['weights'],device='cuda')
        c=torch.arange(65025,device='cuda');self.x=torch.cat([c%255+2/3,c%255+1/3]);self.y=torch.cat([c//255+1/3,c//255+2/3])
        self.q={n:qpoints(n) for n in (3,6)}
    def apply(self,h,tr,tc,n=6):
        ht=torch.as_tensor(np.array(h),device='cuda');g=planes(ht,False);x=tc*255/256;y=(tr+1)*255/256;z=float(np.floor(h[min(255,tr+1),tc])+3)
        q=self.q[n];flux=torch.zeros((130050,6),device='cuda')
        code=self.fn(g.data_ptr(),q.data_ptr(),len(q),x,y,z,float(h.max())+2,flux.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(code)
        alt=g[:,0]+g[:,1]*(self.x%1)+g[:,2]*(self.y%1)+1
        d2=((self.x-x).square()+(self.y-y).square())*(256/255)**2+(alt-z).square()
        direct=flux[:,0]*d2;no=(flux[:,0]+flux[:,1])*d2;yes=(flux[:,0]+flux[:,2])*d2
        vv=torch.stack([direct,no,yes,no-yes,flux[:,3]*(4e9/(4*np.pi)),flux[:,4],flux[:,5],(yes<.4).float(),(no<.4).float(),(yes-no).abs()],1)
        # Interpolate power on the native face, then convert to dB separately.
        pooled=(vv[self.idx]*self.w[:,:,None]).sum(1)
        loss=lambda z:-10*torch.log10(z.clamp(1e-7,1e3))
        out=torch.cat([loss(pooled[:,:3]),(loss(pooled[:,2])-loss(pooled[:,1]))[:,None],
                       torch.log1p(pooled[:,4:6]),pooled[:,6:9],torch.log1p(pooled[:,9:10].clamp_min(0))],1)
        assert torch.isfinite(out).all()
        return out.reshape(256,256,10).cpu().numpy()
    def check(self):
        h=np.zeros((256,256),'float32');v=self.apply(h,125,100)
        y,x=np.indices(h.shape);far=(np.hypot(x-100,y-125)>25)&(y>2)&(y<253)&(x>2)&(x<253)
        err=float(np.quantile(abs(v[:,:,2][far]),.999));assert err<.015 and abs(v[:,:,3]).max()==0,(err,v[:,:,3].min())
        return dict(flat_q999_normalized_dB=err,flat_repeated_power=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['screen','confirm','all'],default='all');ap.add_argument('--shard',type=int,required=True);ap.add_argument('--of',type=int,default=3);ap.add_argument('--check',action='store_true');a=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);engine=Flow();qa=engine.check()
    rows=list(csv.DictReader(open(DATA/'train_index.csv')));h=np.load(DATA/'train_hm.npy',mmap_mode='r')
    for stage in (('screen','confirm') if a.stage=='all' else (a.stage,)):
        data=np.load(A/(stage+'_geo.npz'));ids=data['ids'][a.shard::a.of];px=data['pixels'][a.shard::a.of]
        out=A/(stage+'_flow'+str(a.shard));out.mkdir(exist_ok=True)
        if (out/'done.json').exists() and not a.check:continue
        ar=np.lib.format.open_memmap(out/'field.npy',mode='w+',shape=(len(ids),256,256,10),dtype='float16') if not a.check else None
        tst=[];t=time.monotonic()
        for j,i in enumerate(ids[:1] if a.check else ids):
            row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);hh=h[int(row['hm_row'])]
            y=engine.apply(hh,tr,tc,6)
            if j%12==0:
                z=engine.apply(hh,tr,tc,3);tst.append((int(i),z.reshape(-1,10)[px[j]]))
            if ar is not None:ar[j]=y.astype('float16')
            if j%20==0:print(stage,j+1,len(ids),'seconds',round(time.monotonic()-t,1),'QA',qa,flush=True)
        if a.check:return
        ar.flush();np.savez(out/'q9_control.npz',id=[v[0] for v in tst],lower=np.stack([v[1] for v in tst]))
        atomic_json(out/'done.json',dict(ids=ids.tolist(),gpu=gpu,qa=qa,elapsed=time.monotonic()-t,emit='H and TX only; 36 subfaces primary, paired zero vs Mitsuba-like offset'))
if __name__=='__main__':main()
