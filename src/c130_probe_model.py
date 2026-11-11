"""Critical-obstruction retrieval with matched zero/shuffled-view controls."""
import sys,json,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from src.c125_net import Block
A=R/'audit/c130_projective';S=R/'audit/c101_residual';T=R/'audit/c120_sign'
class Panel:
 def __init__(self,part):
  P=S/part;V=A/'nodes'/part;rows=np.load(V/'rows.npy');assert np.array_equal(rows,np.load(T/(part+'_rows.npy')))
  target=np.load(P/'target.npy',mmap_mode='r')[rows];base=np.load(P/'base.npy',mmap_mode='r')[rows];res=target[:,:2]-base
  self.x=torch.tensor(np.c_[np.load(P/'common.npy',mmap_mode='r')[rows],np.load(P/'precise.npy',mmap_mode='r')[rows]],device='cuda',dtype=torch.float32)
  self.res=torch.tensor(res,device='cuda');self.mask=torch.tensor((target[:,2:]>0)&(np.abs(res)>=5),device='cuda')
  weight=np.load(P/'weight.npy',mmap_mode='r')[rows] if part=='fit' else np.ones(len(rows),np.float32)
  self.weight=torch.tensor(weight/weight.mean(),device='cuda');self.patch=torch.tensor(np.load(V/'patch.npy',mmap_mode='r'),device='cuda')
  self.meta=torch.tensor(np.load(V/'meta.npy'),device='cuda');self.n=len(rows);self.rows=rows;self.part=part
  mapping=np.array([int(r['terrain_id']) for r in csv.DictReader((R.parent/'data/train_index.csv').open())])
  terrain=mapping[np.load(P/'mapid.npy',mmap_mode='r')[rows]];rng=np.random.default_rng(130093 if part=='fit' else 130097);perm=rng.permutation(len(rows))
  for _ in range(100):
   bad=np.flatnonzero(terrain[perm]==terrain)
   if not len(bad):break
   for i in bad:
    for _ in range(1000):
     j=int(rng.integers(len(rows)))
     if terrain[perm[j]]!=terrain[i] and terrain[perm[i]]!=terrain[j]:perm[i],perm[j]=perm[j],perm[i];break
  assert np.all(terrain[perm]!=terrain);self.perm=torch.tensor(perm,device='cuda');self.perm_hash=hashlib.sha256(perm.tobytes()).hexdigest()
  alltarget=np.load(P/'target.npy',mmap_mode='r');res=alltarget[:,:2].astype('float64')-np.load(P/'base.npy',mmap_mode='r');valid=alltarget[:,2:]>0;tail=valid&(np.abs(res)>=5)
  self.total_valid=valid.sum(0);self.nontail_sse=np.where(valid&~tail,res**2,0).sum(0);self.base_sse=np.where(valid,res**2,0).sum(0)
  self.loss_den=(self.mask*self.weight[:,None]).sum(0)/self.n
 def batch(self,ix,mode):
  xx=self.x[ix]
  if mode=='old':patch=torch.zeros_like(self.patch[ix]);meta=torch.zeros_like(self.meta[ix])
  else:
   j=ix if mode=='nodes' else self.perm[ix];patch=self.patch[j];meta=self.meta[j]
  return (xx,patch,meta),self.res[ix],self.mask[ix],self.weight[ix]
class Net(nn.Module):
 def __init__(self,mean,std,mu,prior):
  super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
  self.number_in=nn.Linear(169,256);self.number=nn.Sequential(nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
  self.patch_in=nn.Conv2d(4,32,3,padding=1)
  self.patch=nn.Sequential(nn.SiLU(),nn.Conv2d(32,48,3,2,1),nn.SiLU(),nn.Conv2d(48,64,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(64*5*5,128),nn.LayerNorm(128),nn.SiLU())
  self.meta_in=nn.Linear(8,128);self.role=nn.Parameter(torch.randn(1,3,128)*.02);self.nodes=nn.Sequential(Block(),Block())
  self.attention=nn.MultiheadAttention(128,4,batch_first=True)
  self.mix=nn.Sequential(nn.Linear(256,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
  self.head=nn.Linear(256,4);nn.init.zeros_(self.head.weight)
  with torch.no_grad():self.head.bias[:2].copy_(mu/10);self.head.bias[2:].copy_(torch.logit(prior.clamp(.001,.999)))
 def forward(self,x,patch,meta):
  with torch.autocast('cuda',enabled=False):
   u=self.number_in((x.float()-self.mean)/self.std);p=self.patch_in(patch.float().flatten(0,1));g=self.meta_in(meta.float())
  u=self.number(u);p=self.patch(p).reshape(len(x),3,128);v=self.nodes(p+g+self.role)
  context=self.attention(u[:,None],v,v,need_weights=False)[0][:,0]
  return self.head(self.mix(torch.cat([u,context],1)))
def create(train):
 torch.manual_seed(130123);w=train.mask*train.weight[:,None];mu=(train.res*w).sum(0)/w.sum(0);prior=((train.res>0)*w).sum(0)/w.sum(0)
 return Net(train.x.mean(0),train.x.std(0).clamp_min(.001),mu,prior).cuda()
def loss(net,panel,ix,mode):
 inp,r,valid,w=panel.batch(ix,mode)
 with torch.autocast('cuda',torch.bfloat16):raw=net(*inp)
 raw=raw.float();ww=valid*w[:,None];den=panel.loss_den*len(r)
 mse=((raw[:,:2]-r/10).square()*ww).sum(0)/den
 ce=(F.binary_cross_entropy_with_logits(raw[:,2:],(r>0).float(),reduction='none')*ww).sum(0)/den
 return mse.mean()+.1*ce.mean()
@torch.inference_mode()
def evaluate(net,panel,mode,batch=2048,save=None):
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);count=ss.clone();correct=ss.clone();clcorrect=ss.clone();pred=[];prob=[]
 for st in range(0,panel.n,batch):
  inp,r,m,w=panel.batch(slice(st,st+batch),mode)
  with torch.autocast('cuda',torch.bfloat16):raw=net(*inp)
  delta=10*raw[:,:2].float();err=torch.where(m,delta-r,0);ss+=err.double().square().sum(0);count+=m.sum(0)
  correct+=((delta*r>0)&m).sum(0);clcorrect+=(((raw[:,2:]>0)==(r>0))&m).sum(0)
  if save is not None:pred.append(delta.cpu().numpy());prob.append(raw[:,2:].float().sigmoid().cpu().numpy())
 qrmse=np.sqrt((ss.cpu().numpy()+panel.nontail_sse)/panel.total_valid)
 result=dict(tail_RMSE=(ss/count).sqrt().tolist(),sign_accuracy=(correct/count).tolist(),classification_accuracy=(clcorrect/count).tolist(),
  truth_gated_all_RMSE=qrmse.tolist(),C82_RMSE=np.sqrt(panel.base_sse/panel.total_valid).tolist(),tail_count=count.tolist())
 if save is not None:np.save(str(save)+'_delta.npy',np.concatenate(pred));np.save(str(save)+'_prob.npy',np.concatenate(prob));np.save(str(save)+'_rows.npy',panel.rows)
 net.train();return result
