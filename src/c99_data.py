"""Input-only geometry images and raw TRAIN supervision, with full-precision evaluation targets."""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
from src.c3_geometry import basic_inputs
def mmap(p):return np.load(p,mmap_mode='r',allow_pickle=False)
class Reader:
 def __init__(self):
  self.rows=list(csv.DictReader((D/'train_index.csv').open()));self.h=mmap(D/'train_hm.npy')
  self.y=[mmap(D/f'train_pl{b}.npy') for b in ['415','58']];self.w=[mmap(D/f'train_mask{b}.npy') for b in ['415','58']]
  self.lookup={}
  for folder in ['c73_compact','c73_dev_extra_compact']:
   for p in sorted((R/'features'/folder).glob('part*')):
    if not (p/'complete.json').exists():continue
    ar=mmap(p/'features.npy')
    for pos,i in enumerate(json.loads((p/'indices.json').read_text())):self.lookup[int(i)]=(ar,pos)
  ix=json.loads((R/'audit/c88_mesh/base_index.json').read_text())
  self.c82=mmap(R/'audit/c88_mesh/base.npy');self.c82index={int(i):j for j,i in enumerate(ix['ids'])}
 def fields(self,ids):
  ar,pos=self.lookup[ids[0]]
  if all(self.lookup[i][0] is ar and self.lookup[i][1]==pos+k for k,i in enumerate(ids)):
   return np.array(ar[pos:pos+len(ids)])
  return np.stack([self.lookup[i][0][self.lookup[i][1]] for i in ids])
 def tensors(self,ids):
  to=lambda x:torch.from_numpy(np.ascontiguousarray(x)).cuda()
  row=[self.rows[i] for i in ids]
  h=to(self.h[[int(r['hm_row']) for r in row]]).float();tr=to(np.array([int(r['tx_row']) for r in row]));tc=to(np.array([int(r['tx_col']) for r in row]))
  physical=to(self.fields(ids)).float()
  basic,fs=basic_inputs(h,tr,tc)
  prior=fs+(-20*(physical[:,31:33]+physical[:,41:43])).clamp(-8,48)
  x=torch.cat([basic,physical],1).contiguous(memory_format=torch.channels_last)
  y=to(np.stack([p[ids] for p in self.y],1)).float()
  w=to(np.stack([np.unpackbits(p[ids],axis=1).reshape(-1,256,256) for p in self.w],1)).bool()
  return x,prior,y,w
class Panel:
 def __init__(self,reader,ids,reference=False):
  self.ids=list(ids);n=len(ids);self.reader=reader
  row=[reader.rows[i] for i in ids]
  unique,inv=np.unique([int(r['hm_row']) for r in row],return_inverse=True)
  self.heights=torch.from_numpy(np.array(reader.h[unique])).cuda().float()
  self.hidx=torch.from_numpy(inv).cuda();self.tr=torch.tensor([int(r['tx_row']) for r in row],device='cuda');self.tc=torch.tensor([int(r['tx_col']) for r in row],device='cuda')
  self.physical=torch.empty(n,45,256,256,device='cuda',dtype=torch.float16)
  self.y=torch.empty(n,2,256,256,device='cuda',dtype=torch.float32)
  self.w=torch.empty(n,2,256,256,device='cuda',dtype=torch.bool)
  self.reference=torch.empty(n,2,256,256,device='cuda',dtype=torch.float32) if reference else None
  t=time.monotonic()
  # Preserve logical sample order; read the large physical cache in file order.
  source_order=sorted(range(n),key=lambda j:(str(reader.lookup[ids[j]][0].filename),reader.lookup[ids[j]][1]))
  for start in range(0,n,48):
   dst=source_order[start:start+48];ix=[ids[j] for j in dst]
   index=torch.tensor(dst,device='cuda',dtype=torch.long)
   value=torch.from_numpy(reader.fields(ix)).cuda()
   self.physical.index_copy_(0,index,value)
   del value,index
   if (start//48)%20==0:print('physical_load',start+len(ix),n,'seconds',round(time.monotonic()-t,1),flush=True)
  target_order=sorted(range(n),key=lambda j:ids[j])
  for start in range(0,n,96):
   dst=target_order[start:start+96];ix=[ids[j] for j in dst]
   index=torch.tensor(dst,device='cuda',dtype=torch.long)
   target=torch.from_numpy(np.stack([p[ix] for p in reader.y],1)).cuda().float()
   valid=torch.from_numpy(np.stack([np.unpackbits(p[ix],axis=1).reshape(-1,256,256) for p in reader.w],1)).cuda().bool()
   self.y.index_copy_(0,index,target);self.w.index_copy_(0,index,valid)
   if reference:
    ref=torch.from_numpy(np.stack([reader.c82[reader.c82index[i]] for i in ix])).cuda()
    self.reference.index_copy_(0,index,ref);del ref
   del target,valid,index
  print('panel_loaded',n,'seconds',round(time.monotonic()-t,1),flush=True)
  self.terrain=[reader.rows[i]['terrain_id'] for i in ids]
 def batch(self,ix):
  h=self.heights[self.hidx[ix]];tr=self.tr[ix];tc=self.tc[ix]
  v=self.physical[ix].float();basic,fs=basic_inputs(h,tr,tc)
  prior=fs+(-20*(v[:,31:33]+v[:,41:43])).clamp(-8,48)
  x=torch.cat([basic,v],1).contiguous(memory_format=torch.channels_last)
  return x,prior,self.y[ix],self.w[ix]
