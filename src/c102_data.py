"""Frozen full terrain and point encoders; no target is fed to either encoder."""
import csv,json,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from src.c3_geometry import basic_inputs
from src.c101_c82_cache import lookup,gather
from src.c101_train import Data
from src.c102_transport import frozen_field,frozen_query,geometry
R=Path(__file__).resolve().parents[1]
class Bank:
 def __init__(self,part,source_net,limit=0):
  self.points=Data(part,'coarse');self.meta=self.points.meta;self.ids=self.meta['ids'][:limit] if limit else self.meta['ids'];self.pp=self.meta['points_per_map'];n=len(self.ids)
  root=R/'audit/c101_residual'/part
  self.precise=torch.from_numpy(np.array(np.load(root/'precise.npy',mmap_mode='r'))).cuda()
  self.pixels=torch.from_numpy(np.array(np.load(root/'pixel.npy',mmap_mode='r').reshape(-1,self.pp)[:n],np.int64)).cuda()
  rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));hh=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
  unique,inv=np.unique([int(rows[i]['hm_row']) for i in self.ids],return_inverse=True)
  self.heights=torch.from_numpy(np.array(hh[unique])).cuda();self.hidx=torch.tensor(inv,device='cuda')
  self.tr=torch.tensor([int(rows[i]['tx_row']) for i in self.ids],device='cuda');self.tc=torch.tensor([int(rows[i]['tx_col']) for i in self.ids],device='cuda')
  self.physical=torch.empty(n,45,256,256,device='cuda',dtype=torch.float16)
  self.c82=torch.empty(n,2,256,256,device='cuda',dtype=torch.float32)
  compact=lookup(['features/c73_compact/part*','features/c73_dev_extra_compact/part*'],'features.npy')
  if part=='fit':
   base={}
   for directory in (R/'audit/c101_residual').glob('base*of*'):
    if not (directory/'done.json').exists():continue
    ar=np.load(directory/'prediction.npy',mmap_mode='r')
    base.update({int(i):(ar,j) for j,i in enumerate(json.loads((directory/'indices.json').read_text()))})
  else:
   meta=json.loads((R/'audit/c88_mesh/base_index.json').read_text());ar=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
   base={int(i):(ar,j) for j,i in enumerate(meta['ids'])}
  order=sorted(range(n),key=lambda j:(str(compact[self.ids[j]][0].filename),compact[self.ids[j]][1]))
  t=time.monotonic()
  for start in range(0,n,48):
   dst=order[start:start+48];ids=[self.ids[j] for j in dst];index=torch.tensor(dst,device='cuda')
   self.physical.index_copy_(0,index,torch.from_numpy(gather(compact,ids)).cuda())
   self.c82.index_copy_(0,index,torch.from_numpy(gather(base,ids)).cuda())
  self.features=torch.empty(n,48,256,256,device='cuda',dtype=torch.float16)
  for start in range(0,n,8):
   ix=torch.arange(start,min(n,start+8),device='cuda');x,_,_,_=self.image(ix)
   with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):v=frozen_field(source_net,x)
   self.features[ix]=v.half()
  check=torch.arange(min(4,n),device='cuda');frommap=self.c82[check].flatten(2).gather(2,self.pixels[check,None].expand(-1,2,-1)).transpose(1,2)
  assert torch.equal(frommap,self.points.base.reshape(-1,self.pp,2)[check])
  assert all(bool(torch.isfinite(chunk).all()) for chunk in self.features.split(8))
  rng=np.random.default_rng(102017)
  self.monitor_positions=torch.tensor(np.stack([rng.choice(self.pp,256,replace=False) for _ in self.ids]),device='cuda')
  print('BANK',part,n,'seconds',round(time.monotonic()-t,1),'allocatedGiB',torch.cuda.memory_allocated()/2**30,flush=True)
 def image(self,maps):
  h=self.heights[self.hidx[maps]];physical=self.physical[maps].float();basic,fs=basic_inputs(h,self.tr[maps],self.tc[maps])
  return torch.cat([basic,physical],1).contiguous(memory_format=torch.channels_last),h,physical,fs
 @torch.no_grad()
 def batch(self,maps,positions,scale,point_net,need_source=True):
  indices=maps[:,None]*self.pp+positions;flat=indices.reshape(-1)
  coarse=torch.cat([self.points.common[flat],self.points.extra[flat]],1)
  precise=torch.cat([self.points.common[flat],self.precise[flat]],1)
  with torch.autocast('cuda',torch.bfloat16):query,delta=frozen_query(point_net,coarse,precise)
  b,q=positions.shape;query=query.reshape(b,q,681);baseline=self.points.base[indices]+delta.reshape(b,q,2)
  targets=self.points.target[indices];weight=self.points.weight[indices]
  if not need_source:return query,None,None,None,baseline,targets,weight
  x,h,compact,fs=self.image(maps)
  values=torch.cat([self.features[maps].float(),x,(self.c82[maps]-fs)/20],1)
  if scale>1:values=F.avg_pool2d(values,scale)
  nodes=values.flatten(2).transpose(1,2).contiguous()
  pixels=self.pixels[maps[:,None],positions]
  bias,mass=geometry(h,self.tr[maps],self.tc[maps],pixels,compact,scale)
  return query,nodes,bias,mass,baseline,targets,weight
