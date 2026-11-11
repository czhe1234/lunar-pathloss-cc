"""Matched raw-view and compressed-input tail learners."""
import sys,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
A=R/'audit/c125_tail';S=R/'audit/c101_residual';T=R/'audit/c120_sign'
class Block(nn.Module):
 def __init__(self,d=128):
  super().__init__();self.norm=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
 def forward(self,x):
  b,t,d=x.shape;q,k,v=self.qkv(self.norm(x)).reshape(b,t,3,4,d//4).permute(2,0,3,1,4).unbind(0)
  z=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,t,d)
  x=x+self.proj(z);return x+self.ff(x)
class Net(nn.Module):
 def __init__(self,mean,std,prior,mu):
  super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
  self.number_in=nn.Linear(169,256);self.number=nn.Sequential(nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
  self.rx_in=nn.Conv2d(2,32,3,padding=1)
  self.rx=nn.Sequential(nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,96,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(96*9*9,128),nn.LayerNorm(128),nn.SiLU())
  self.tx_in=nn.Conv2d(2,16,3,padding=1)
  self.tx=nn.Sequential(nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.Conv2d(32,48,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(48*5*5,64),nn.LayerNorm(64),nn.SiLU())
  self.path_in=nn.Conv1d(9,128,3,padding=1);self.position=nn.Parameter(torch.randn(1,129,128)*.02);self.path=nn.Sequential(Block(),Block())
  self.query=nn.Linear(320,128);self.attn=nn.MultiheadAttention(128,4,batch_first=True)
  self.mix=nn.Sequential(nn.Linear(448,384),nn.LayerNorm(384),nn.SiLU(),nn.Linear(384,256),nn.SiLU())
  self.head=nn.Linear(256,4);nn.init.zeros_(self.head.weight)
  with torch.no_grad():self.head.bias[:2].copy_(mu/10);self.head.bias[2:].copy_(torch.logit(prior.clamp(.001,.999)))
 def forward(self,x,rx,tx,path):
  # Preserve centimetre-scale input differences in the first projection.
  with torch.autocast('cuda',enabled=False):
   u=self.number_in((x.float()-self.mean)/self.std)
   a=self.rx_in(rx.float());b=self.tx_in(tx.float());p=self.path_in(path.float()).transpose(1,2)
  u=self.number(u);a=self.rx(a);b=self.tx(b);p=self.path(p+self.position)
  z=torch.cat([u,a,b],1);q=self.query(z)[:,None]
  att=self.attn(q,p,p,need_weights=False)[0][:,0]
  return self.head(self.mix(torch.cat([z,att],1)))
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
class Panel:
 def __init__(self,part):
  p=S/part;v=A/'views'/part;assert (v/'done.json').exists()
  rows=np.load(v/'rows.npy');assert np.array_equal(rows,np.load(T/(part+'_rows.npy')))
  target=np.load(p/'target.npy',mmap_mode='r')[rows];base=np.load(p/'base.npy',mmap_mode='r')[rows];res=target[:,:2]-base
  x=np.c_[np.load(p/'common.npy',mmap_mode='r')[rows],np.load(p/'precise.npy',mmap_mode='r')[rows]].astype('float32')
  self.x=torch.tensor(x,device='cuda');self.res=torch.tensor(res,device='cuda');self.mask=torch.tensor((target[:,2:4]>0)&(np.abs(res)>=5),device='cuda')
  w=np.load(p/'weight.npy',mmap_mode='r')[rows].copy() if part=='fit' else np.ones(len(rows),np.float32)
  self.weight=torch.tensor(w/w.mean(),device='cuda')
  self.view=[torch.tensor(np.load(v/(key+'.npy')),device='cuda') for key in ['rx','tx','corridor']]
  self.rows=rows;self.n=len(rows);self.part=part
  # A fixed permutation with no same-terrain donor. No residual values enter this permutation.
  maps=np.load(p/'mapid.npy',mmap_mode='r')[rows];csvrows=__import__('csv').DictReader(open(R.parent/'data/train_index.csv'));termap=np.array([int(r['terrain_id']) for r in csvrows])
  terrain=termap[maps];rng=np.random.default_rng(125093 if part=='fit' else 125097);perm=rng.permutation(len(rows))
  for _ in range(100):
   bad=np.flatnonzero(terrain[perm]==terrain)
   if not len(bad):break
   for i in bad:
    for _ in range(1000):
     j=int(rng.integers(len(rows)))
     if terrain[perm[j]]!=terrain[i] and terrain[perm[i]]!=terrain[j]:perm[i],perm[j]=perm[j],perm[i];break
  assert np.all(terrain[perm]!=terrain)
  self.perm=torch.tensor(perm,device='cuda');self.perm_hash=hashlib.sha256(perm.tobytes()).hexdigest()
  rr=np.load(p/'target.npy',mmap_mode='r')[:,:2].astype('float64')-np.load(p/'base.npy',mmap_mode='r')
  mm=np.load(p/'target.npy',mmap_mode='r')[:,2:4]>0;tt=(np.abs(rr)>=5)&mm
  self.n_all=mm.sum(0);self.nontail_sse=np.where(mm&~tt,rr**2,0).sum(0);self.base_sse=np.where(mm,rr**2,0).sum(0)
 def batch(self,ix,mode):
  x=self.x[ix]
  if mode=='old':v=[torch.zeros_like(a[ix]) for a in self.view]
  elif mode=='shuffled':v=[a[self.perm[ix]] for a in self.view]
  else:v=[a[ix] for a in self.view]
  return (x,*v),self.res[ix],self.mask[ix],self.weight[ix]
def loss(model,panel,ix,mode):
 inp,r,m,w=panel.batch(ix,mode)
 with torch.autocast('cuda',torch.bfloat16):o=model(*inp)
 o=o.float();ww=m.float()*w[:,None];den=ww.sum(0).clamp_min(1)
 mse=((o[:,:2]-r/10).square()*ww).sum(0)/den
 ce=(F.binary_cross_entropy_with_logits(o[:,2:],(r>0).float(),reduction='none')*ww).sum(0)/den
 return mse.mean()+.1*ce.mean(),o
@torch.inference_mode()
def evaluate(model,panel,mode,batch=512,indices=None,save=None):
 model.eval();n=panel.n if indices is None else len(indices);s=torch.zeros(2,device='cuda',dtype=torch.float64);c=s.clone();correct=s.clone();classcorrect=s.clone();energycorrect=s.clone();energy=s.clone();predictions=[];classpred=[]
 for st in range(0,n,batch):
  ix=slice(st,min(st+batch,n)) if indices is None else indices[st:st+batch]
  inp,r,m,w=panel.batch(ix,mode)
  with torch.autocast('cuda',torch.bfloat16):raw=model(*inp)
  d=raw[:,:2].float()*10;sign=raw[:,2:].float()>0;e=torch.where(m,d-r,0.)
  s+=e.double().square().sum(0);c+=m.sum(0);correct+=(((d>=0)==(r>0))&m).sum(0);classcorrect+=((sign==(r>0))&m).sum(0)
  e2=torch.where(m,r.double().square(),0.);energy+=e2.sum(0);energycorrect+=(e2*((d>=0)==(r>0))).sum(0)
  if save is not None:predictions.append(d.cpu().numpy());classpred.append(raw[:,2:].float().sigmoid().cpu().numpy())
 out=dict(tail_RMSE=(s/c).sqrt().tolist(),sign_accuracy=(correct/c).tolist(),classification_accuracy=(classcorrect/c).tolist(),squared_residual_weighted_sign_accuracy=(energycorrect/energy).tolist(),n=c.tolist())
 if indices is None:
  orrm=np.sqrt((panel.nontail_sse+s.cpu().numpy())/panel.n_all);out['truth_gated_all_RMSE']=orrm.tolist();out['truth_gated_gain_vs_C82']=(np.sqrt(panel.base_sse/panel.n_all)-orrm).tolist()
 if save is not None:np.save(str(save)+'_delta.npy',np.concatenate(predictions));np.save(str(save)+'_prob.npy',np.concatenate(classpred));np.save(str(save)+'_rows.npy',panel.rows)
 model.train();return out
def create(train):
 mean=train.x.mean(0);std=train.x.std(0).clamp_min(.001);ww=train.mask*train.weight[:,None];mu=(ww*train.res).sum(0)/ww.sum(0);prior=(ww*(train.res>0)).sum(0)/ww.sum(0)
 torch.manual_seed(125123)
 return Net(mean,std,prior,mu).cuda()
