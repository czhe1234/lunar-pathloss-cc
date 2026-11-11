"""CPU-resident release labels, GPU terrain, input geometry cached separately."""
import csv,json,time
import numpy as np
import torch
from cc_core import ROOT,DATA
class TrainingData:
    def __init__(self,split='train'):
        self.split=split;self.rows=list(csv.DictReader((DATA/f'{split}_index.csv').open()))
        self.h=torch.from_numpy(np.array(np.load(DATA/f'{split}_hm.npy',mmap_mode='r'),copy=True)).cuda()
        self.hmrow=torch.tensor([int(r['hm_row']) for r in self.rows],device='cuda');self.txr=torch.tensor([int(r['tx_row']) for r in self.rows],device='cuda');self.txc=torch.tensor([int(r['tx_col']) for r in self.rows],device='cuda')
        n=len(self.rows);self.y=torch.empty((n,2,256,256),dtype=torch.float16);self.m=torch.empty((n,2,256,256),dtype=torch.uint8)
        for k,band in enumerate(['415','58']):
            yy=np.load(DATA/f'{split}_pl{band}.npy',mmap_mode='r');mm=np.load(DATA/f'{split}_mask{band}.npy',mmap_mode='r')
            for lo in range(0,n,512):
                hi=min(lo+512,n);self.y[lo:hi,k].copy_(torch.from_numpy(np.array(yy[lo:hi],copy=True)));self.m[lo:hi,k].copy_(torch.from_numpy(np.unpackbits(mm[lo:hi],axis=1,bitorder='big').reshape(hi-lo,256,256)))
        self.buffer_y=torch.empty((64,2,256,256),dtype=torch.float16,pin_memory=True);self.buffer_m=torch.empty((64,2,256,256),dtype=torch.uint8,pin_memory=True);self.copy_event=None
    def indices(self,part):
        if part=='full':return np.arange(len(self.rows),dtype=np.int64)
        dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids'])
        return np.array([i for i,r in enumerate(self.rows) if (r['terrain_id'] in dev)==(part=='dev')],dtype=np.int64)
    def targets(self,indices):
        if self.copy_event is not None:self.copy_event.synchronize()
        ids=torch.as_tensor(np.asarray(indices),dtype=torch.long);n=len(ids);assert n<=64
        torch.index_select(self.y,0,ids,out=self.buffer_y[:n]);torch.index_select(self.m,0,ids,out=self.buffer_m[:n])
        y=self.buffer_y[:n].to('cuda',non_blocking=True).float();m=self.buffer_m[:n].to('cuda',non_blocking=True).float()
        if self.copy_event is None:self.copy_event=torch.cuda.Event()
        self.copy_event.record()
        return y,m
