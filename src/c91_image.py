"""Full facet search for specular image sources on candidate TRAIN terrain only."""
import argparse,ctypes,csv,json,time,sys
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c67_physics import quadrature
from src.c57_transmission import slab
from src.c69_registered import RegisteredTransport
A=R/'audit/c91_image'
class Image:
    def __init__(self):
        self.rt=RegisteredTransport(physical=1,anti=0,smooth=1,direct=1,spec=1,diffuse=1)
        self.q=quadrature(2);self.lib=ctypes.CDLL(str(R/'src/libc91_image.so'))
        p=ctypes.c_void_p;i=ctypes.c_int;s=ctypes.c_float;self.fn=self.lib.image91
        self.fn.argtypes=[p,p,p,p,i,s,s,s,s,i,p,p,p];self.fn.restype=i
    def trace(self,g,ns,x,y,z,top,rx,mode):
        n=len(rx);im=torch.empty_like(g);out=torch.zeros((n,14),device='cuda')
        err=self.fn(g.data_ptr(),ns.data_ptr(),self.rt.material.data_ptr(),rx.data_ptr(),n,x,y,z,top,mode,im.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if err:raise RuntimeError(err)
        return out
    def check(self):
        h=torch.zeros((256,256),device='cuda');g,ns=self.rt.scene(h)
        xy=torch.tensor([[93.461,75.123],[41.247,190.836],[215.571,61.284]],device='cuda')
        rx=torch.cat([xy,torch.ones_like(xy[:,:1])],1).contiguous()
        dx=(xy[:,0]-105.25)*self.rt.spacing;dy=(xy[:,1]-109.64)*self.rt.spacing
        d2=dx.square()+dy.square()+4;rl2=dx.square()+dy.square()+16
        _,_,_,rp=slab(4/rl2.sqrt(),torch.tensor([415e6,5.775e9],device='cuda'),torch.tensor([.0003562,.009324],device='cuda'))
        expected=(1-.632**2)*rp.abs().square()*(d2/rl2)[:,None]
        result=[]
        for mode in (0,1):
            v=self.trace(g,ns,105.25,109.64,3.,0.,rx,mode)
            assert (v[:,0:3]==1).all(),v
            rel=((v[:,3:5]-expected).abs()/expected.clamp_min(.0001)).max().item()
            assert rel<.002,rel;result.append(rel)
        return dict(flat_single_visible_reflection=True,relative_power_error=result)
    def features(self,h,r,c,px):
        ht=torch.as_tensor(np.array(h),device='cuda');g,norm=self.rt.scene(ht)
        ii=self.rt.ids[torch.as_tensor(px,device='cuda')];w=self.rt.weights[torch.as_tensor(px,device='cuda')]
        uf,inv=torch.unique(ii,return_inverse=True)
        verts=torch.cat([self.rt.xy/self.rt.spacing,ht.reshape(-1,1)+1],1)[self.rt.faces[uf]]
        rx=torch.einsum('qc,fcv->fqv',self.q,verts).reshape(-1,3).contiguous()
        z=self.rt.source_height(h,r,c);top=float(h.max())
        allv=[]
        for mode in (0,1):
            v=self.trace(g,norm,c*255/256.,(r+1)*255/256.,z,top,rx,mode).reshape(len(uf),len(self.q),14)
            weight=w[:,:,None,None]/len(self.q);vals=v[inv];avg=(vals*weight).sum((1,2))
            def loss(s):return -10*torch.log10(s.clamp(1e-13,1e8))
            count=torch.log1p(avg[:,:3])
            power=loss(avg[:,3:9]);strength=avg[:,[6,6,5,5]].clamp_min(1e-14)
            path=(avg[:,9:13]/strength).clamp(0,15)
            eff=(avg[:,6].square()/avg[:,13].clamp_min(1e-20)).clamp(0,80)
            hit=((vals[:,:,:,2]>0)*weight[:,:,:,0]).sum((1,2))
            spread=(((vals[:,:,:,6]>1e-7)*weight[:,:,:,0]).sum((1,2)))
            allv.append(torch.cat([count,power,path,eff[:,None],hit[:,None],spread[:,None]],1))
        out=torch.cat(allv,1);assert out.shape[1]==32 and torch.isfinite(out).all()
        return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=3)
    p.add_argument('--check',action='store_true');p.add_argument('--stage',choices=['screen','confirm'],default='screen');a=p.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);e=Image();qa=e.check()
    panel=json.load(open(R/'audit/c79_receiver/protocol.json'))['panel']
    panel=[r for r in panel if (r['group']=='confirm')==(a.stage=='confirm')][a.shard::a.of]
    pixels=np.load(R/'audit/c78_clear/train_panel.npz')['pixels'];rows=list(csv.DictReader(open(DATA/'train_index.csv')));h= np.load(DATA/'train_hm.npy',mmap_mode='r')
    out=A/(a.stage+str(a.shard));out.mkdir(exist_ok=True);ts=time.monotonic()
    rec=panel[:1] if a.check else panel
    if not a.check:ar=np.lib.format.open_memmap(out/'x.npy',mode='w+',shape=(len(rec),512,32),dtype='float16')
    locs=[]
    for j,r in enumerate(rec):
        i=r['index'];row=rows[i];loc=np.sort(np.random.default_rng([91037,i]).choice(2048,512,replace=False));locs.append(loc)
        v=e.features(h[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']),pixels[r['pos'],loc].astype('int64')).half().cpu().numpy()
        if not a.check:ar[j]=v
        if j%15==0:print(j+1,'/',len(rec),'elapsed',round(time.monotonic()-ts,2),flush=True)
    if a.check:
        print('QA',qa,'seconds for one map',time.monotonic()-ts,flush=True);return
    ar.flush();np.save(out/'loc.npy',np.array(locs));atomic_json(out/'done.json',dict(rows=rec,gpu=gpu,qa=qa,elapsed=time.monotonic()-ts))
if __name__=='__main__':main()
