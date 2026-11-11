"""Released TRAIN-only resident data for C43; labels/masks remain separate."""
from pathlib import Path
import csv,time,json
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
class Data:
    def __init__(self,ids):
        start=time.time();self.rows=list(csv.DictReader((D/'train_index.csv').open()))
        self.ids=np.array(sorted(set(map(int,ids))),np.int64);n=len(self.ids)
        hr=np.array([int(self.rows[i]['hm_row']) for i in self.ids])
        hs,hmrow=np.unique(hr,return_inverse=True)
        H=np.load(D/'train_hm.npy',mmap_mode='r')
        self.h=torch.from_numpy(np.array(H[hs],np.float32)).cuda()
        self.hmrow=torch.tensor(hmrow,device='cuda')
        self.txr=torch.tensor([int(self.rows[i]['tx_row']) for i in self.ids],device='cuda')
        self.txc=torch.tensor([int(self.rows[i]['tx_col']) for i in self.ids],device='cuda')
        self.lookup=torch.full((len(self.rows),),-1,dtype=torch.long,device='cuda');self.lookup[torch.tensor(self.ids,device='cuda')]=torch.arange(n,device='cuda')
        self.y=torch.empty((n,2,256,256),dtype=torch.float16,device='cuda')
        self.m=torch.empty((n,2,256,256),dtype=torch.uint8,device='cuda')
        for b,band in enumerate(['415','58']):
            y=np.load(D/f'train_pl{band}.npy',mmap_mode='r');m=np.load(D/f'train_mask{band}.npy',mmap_mode='r')
            self.y[:,b].copy_(torch.from_numpy(np.array(y[self.ids])))
            mm=np.unpackbits(np.array(m[self.ids]),axis=1,bitorder='big').reshape(n,256,256)
            self.m[:,b].copy_(torch.from_numpy(mm))
        self.yy,self.xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij')
        print(json.dumps({'event':'c43_data','maps':n,'terrains':len(hs),'seconds':time.time()-start,'GPU_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
    def batch(self,indices):
        ii=torch.as_tensor(indices,dtype=torch.long,device='cuda');j=self.lookup[ii]
        assert bool((j>=0).all())
        h=self.h[self.hmrow[j]];r=self.txr[j];c=self.txc[j];B=len(j)
        ht=h[torch.arange(B,device='cuda'),r,c]
        dx=self.xx[None]-c[:,None,None];dy=self.yy[None]-r[:,None,None]
        dz=h+1-ht[:,None,None]-3;dist=torch.sqrt(dx*dx+dy*dy+dz*dz).clamp_min(.25)
        sy,sx=torch.gradient(h,dim=(-2,-1))
        from torch.nn import functional as F
        hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0]
        x=torch.stack((h/100,(h-ht[:,None,None])/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(dist)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx*dx+dy*dy)/8)),1).contiguous(memory_format=torch.channels_last)
        tx=torch.stack((c.float(),r.float(),ht+3),1)
        freq=torch.tensor([415e6,5.775e9],device='cuda')
        fs=20*torch.log10(4*torch.pi*dist[:,None]*freq[None,:,None,None]/299792458.)
        return x,tx,fs,self.y[j],self.m[j]
def gather(a,p):
    return a.flatten(2).gather(2,p[:,None].expand(-1,2,-1)).transpose(1,2)
