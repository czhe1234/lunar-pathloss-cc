"""Registered exact terrain profiles. No material coefficients or simulated powers."""
import argparse,ctypes,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,atomic_json,DATA
import torch
import numpy as np

def qpoints(n):
    values=[]
    for i in range(n):
        for j in range(n-i):
            for a in ((1/3,2/3) if i+j+1<n else (1/3,)):
                u=(i+a)/n;v=(j+a)/n;values.append([1-u-v,u,v])
    return torch.tensor(values,dtype=torch.float32,device='cuda')

class GeometricMaps:
    def __init__(self,root=None,anti=0):
        self.anti=anti;root=Path(root or ROOT/'src');rec=np.load(root/f'c70_centroid_lookup{anti}.npz')
        self.ids=torch.tensor(rec['ids'],dtype=torch.long,device='cuda')
        self.w=torch.tensor(rec['weights'],dtype=torch.float32,device='cuda')
        self.quad=qpoints(3);self.lib=ctypes.CDLL(str(root/'libc70_geometry.so'))
        P=ctypes.c_void_p;I=ctypes.c_int;S=ctypes.c_float
        self.fn=self.lib.ground_geometry;self.fn.argtypes=[P,P,I,S,S,S,S,I,P,P];self.fn.restype=I
    @torch.no_grad()
    def native(self,h,tr,tc):
        a=h[:-1,:-1].flatten();b=h[:-1,1:].flatten();c=h[1:,:-1].flatten();d=h[1:,1:].flatten()
        if self.anti:
            z=torch.cat((a,b+c-d));sx=torch.cat((b-a,d-c));sy=torch.cat((c-a,d-b))
        else:
            z=torch.cat((a,a));sx=torch.cat((b-a,d-c));sy=torch.cat((d-b,c-a))
        spacing=256/255;s=torch.sqrt(1+(sx/spacing)**2+(sy/spacing)**2)*(.5*spacing**2)
        g=torch.stack((z,sx,sy,s),1).contiguous();out=torch.empty((2*255*255,27),device='cuda',dtype=torch.float32)
        ztx=float(h[min(tr+1,255),tc].floor().item()+3);x=tc*255/256;y=(tr+1)*255/256
        code=self.fn(g.data_ptr(),self.quad.data_ptr(),len(self.quad),x,y,ztx,spacing,self.anti,out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(code)
        return out,ztx
    @torch.no_grad()
    def features(self,h,tr,tc):
        s=torch.as_tensor(np.array(h,copy=True),device='cuda',dtype=torch.float32) if not torch.is_tensor(h) else h
        native,ztx=self.native(s,tr,tc)
        val=native[self.ids];w=self.w[:,:,None]
        mean=(val*w).sum(1)
        sel=[8,9,10,14,18,19,20]
        std=((val[:,:,sel]-mean[:,None,sel]).square()*w).sum(1).clamp_min(0).sqrt()
        # The largest two face weights and native LOS soft labels preserve unresolved receiver-cell structure.
        ord=self.w.argsort(1,descending=True);ip=torch.arange(len(ord),device='cuda')
        w0=self.w[ip,ord[:,0]];w1=self.w[ip,ord[:,1]]
        v0=val[ip,ord[:,0],18];v1=val[ip,ord[:,1],18]
        zdelta=ztx-3-s[tr,tc];df=torch.ones_like(w0)*zdelta
        vals=torch.cat((mean,std,torch.stack((w0,w1,v0,v1,df),1)),1)
        return vals.T.reshape(-1,256,256).contiguous()
def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--dir',default='features/c70_geometry');p.add_argument('--panel',default='')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1)
    a=p.parse_args();gpu=gpu_guard();eng=GeometricMaps()
    if a.check:
        h=torch.zeros((256,256),device='cuda');v,z=eng.native(h,120,130)
        assert (v[:,18]>.999).all() and (v[:,14]==0).all() and torch.allclose(v[:,8],torch.full_like(v[:,8],float(np.arcsinh(-4)/3)),atol=2e-4)
        h=(12-(torch.arange(256,device='cuda')-128).abs()).clamp_min(0).float()[None].expand(256,256)
        v,z=eng.native(h,100,100)
        f=100*255+158;assert abs(float(v[f,14])-np.log(3)/2)<.0001 and float(v[f,18])==0
        assert torch.isfinite(eng.features(h,100,100)).all()
        print('flat and ridge reference checks passed',flush=True);return
    d=json.load(open(ROOT/'audit/c66_transport/panels.json'))
    ix=json.load(open(ROOT/a.panel)) if a.panel else d['dev_maps']+d['fit_maps'];ix=ix[a.shard::a.of]
    dest=ROOT/a.dir/f'part{a.shard}of{a.of}';dest.mkdir(parents=True,exist_ok=False);atomic_json(dest/'indices.json',ix)
    arr=np.lib.format.open_memmap(dest/'features.npy','w+',dtype=np.float16,shape=(len(ix),39,256,256))
    rows=list(csv.DictReader((DATA/'train_index.csv').open()));hh=np.load(DATA/'train_hm.npy',mmap_mode='r')
    t=time.monotonic()
    for j,i in enumerate(ix):
        r=rows[i];val=eng.features(hh[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col'])).half().cpu().numpy()
        if not np.isfinite(val).all():raise RuntimeError(f'Nonfinite geometry at {i}')
        arr[j]=val
        if j%50==0 or j+1==len(ix):
            prog={'maps':j+1,'total':len(ix),'elapsed':round(time.monotonic()-t,1)};atomic_json(dest/'progress.json',prog);print(prog,flush=True)
    arr.flush();atomic_json(dest/'complete.json',dict(vars(a),gpu=gpu))
if __name__=='__main__':main()
