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
        self.y=[np.load(DATA/f'{split}_pl{b}.npy',mmap_mode='r') for b in ['415','58']];self.m=[np.load(DATA/f'{split}_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
    def indices(self,part):
        if part=='full':return np.arange(len(self.rows),dtype=np.int64)
        dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids'])
        return np.array([i for i,r in enumerate(self.rows) if (r['terrain_id'] in dev)==(part=='dev')],dtype=np.int64)
    def targets(self,indices):
        ids=np.asarray(indices);y=np.stack([a[ids] for a in self.y],1)
        mask=np.stack([np.unpackbits(a[ids],axis=1,bitorder='big').reshape(len(ids),256,256) for a in self.m],1)
        return torch.from_numpy(y).cuda().float(),torch.from_numpy(mask).cuda().float()
