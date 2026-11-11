"""C97 bounded information-organization control on CPU; no CUDA initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,json,time,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1];A=R/'audit/c97_profile';D=R/'audit/c96_information'
def dump(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
class Block(nn.Module):
 def __init__(self,d):
  super().__init__();self.norm=nn.LayerNorm(d);self.fc=nn.Sequential(nn.Linear(d,d*2),nn.SiLU(),nn.Linear(d*2,d))
 def forward(self,x):return x+.25*self.fc(self.norm(x))
class ProfileBlock(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(32);self.conv=nn.Conv1d(32,32,5,padding=2,groups=32);self.fc=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,32))
 def forward(self,x):
  y=self.conv(self.norm(x).transpose(1,2)).transpose(1,2)
  return x+.25*self.fc(y)
class Net(nn.Module):
 def __init__(self,variant,mean,std):
  super().__init__();self.variant=variant
  self.register_buffer('mean',mean);self.register_buffer('std',std)
  self.g=nn.Sequential(nn.Linear(296,128),nn.SiLU(),Block(128),Block(128))
  self.node=nn.Linear(12,32);self.p=nn.Sequential(ProfileBlock(),ProfileBlock())
  self.query=nn.Linear(128,64);self.key=nn.Linear(32,32);self.value=nn.Linear(32,32)
  self.head=nn.Sequential(nn.Linear(286,128),nn.SiLU(),Block(128),nn.Linear(128,2))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
  kind=torch.cat([torch.full((n,),k,dtype=torch.long) for k,n in enumerate([64,24,24])])
  self.register_buffer('kind',F.one_hot(kind,3).float())
 def forward(self,x,s):
  z=self.g(((x-self.mean)/self.std).clamp(-15,15))
  raw=s[:,:672].reshape(-1,6,112)
  if self.variant=='mean':raw=raw.mean(1,keepdim=True).expand(-1,6,-1)
  dx=x[:,5]*128;dy=x[:,6]*128;horizontal=torch.sqrt(dx*dx+dy*dy).clamp_min(1.)
  central=(torch.arange(64,device=x.device)+.5)/64
  near=(torch.arange(24,device=x.device)+1)[None]/horizontal[:,None]
  t=torch.cat([central[None].expand(len(x),-1),near.clamp(0,1),(1-near).clamp(0,1)],1)
  position=torch.stack([t,torch.log1p(t*horizontal[:,None])/6,torch.log1p((1-t)*horizontal[:,None])/6],2)
  nodes=self.node(torch.cat([raw.transpose(1,2),position,self.kind[None].expand(len(x),-1,-1)],2))
  encoded=torch.cat([self.p(v) for v in nodes.split([64,24,24],1)],1)
  q=self.query(z).reshape(-1,2,32)
  scores=torch.bmm(q,self.key(encoded).transpose(1,2))/math.sqrt(32)
  context=torch.bmm(scores.softmax(-1),self.value(encoded)).flatten(1)
  out=self.head(torch.cat([z,context,encoded.mean(1),encoded.amax(1),s[:,672:]],1))
  return out
def data(part,which=None):
 p=D/part
 keys=['x','strip','y','mask','base','row_weight']
 out={}
 for k in keys:
  a=np.load(p/(k+'.npy'),mmap_mode='r')
  out[k]=torch.from_numpy(np.array(a if which is None else a[which],np.float32))
 return out
@torch.inference_mode()
def evaluate(net,d,batch=2048):
 net.eval();ss=torch.zeros(2,dtype=torch.float64);cc=torch.zeros(2,dtype=torch.float64)
 for lo in range(0,len(d['x']),batch):
  sl=slice(lo,lo+batch);p=d['base'][sl]+20*net(d['x'][sl],d['strip'][sl])
  e=(p-d['y'][sl]).double();m=d['mask'][sl].double();ss+=(e.square()*m).sum(0);cc+=m.sum(0)
 net.train();return dict(rmse=(ss/cc).sqrt().tolist(),sse=ss.tolist(),count=cc.tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['mean','strip'],required=True);ap.add_argument('--threads',type=int,default=12)
 ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(970431)
 A.mkdir(exist_ok=True);out=A/a.variant;out.mkdir(exist_ok=True)
 assert a.smoke or not (out/'result.json').exists()
 dm=json.loads((D/'dev/done.json').read_text());sel=np.repeat(np.array(dm['groups'])!='confirm',256)
 fit=data('fit');dev=data('dev',sel)
 mean=fit['x'].mean(0);std=fit['x'].std(0).clamp_min(1e-3)
 model=Net(a.variant,mean,std);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
 generator=torch.Generator().manual_seed(970439);t=time.monotonic()
 if a.smoke:
  for batch in [128,512,2048]:
   timing=[]
   for k in range(12):
    st=time.monotonic();j=torch.randint(len(fit['x']),(batch,),generator=generator)
    opt.zero_grad(set_to_none=True);q=model(fit['x'][j],fit['strip'][j])
    target=(fit['y'][j]-fit['base'][j])/20;m=fit['mask'][j]
    loss=((q-target).square()*m).sum()/m.sum().clamp_min(1);loss.backward();opt.step()
    assert torch.isfinite(loss);timing.append(time.monotonic()-st)
   print('benchmark',batch,'seconds_per_step',float(np.mean(timing[3:])),flush=True)
  return
 cfg=dict(**vars(a),CPU=True,seed=970431,parameters=sum(p.numel() for p in model.parameters()),fit_points=len(fit['x']),monitor_points=len(dev['x']),
  architecture='H/TX conditional profile convolution plus band queries; frozen C38 residual base',
  only_difference='mean averages six lateral raw profiles and repeats them; strip preserves all six. Both retain identical old296 features and30strip summaries.',
  selection='joint mean of masked two-band monitorRMSE, includes zero correction; patience300 updates, evaluate100',
  confirmation_labels_used_in_selection=False,official_val_used=False,torch=torch.__version__)
 dump(out/'config.json',cfg)
 initial=evaluate(model,dev);best=float(np.mean(initial['rmse']));best_step=0
 torch.save(dict(model=model.state_dict(),config=cfg,step=0),out/'best.pt')
 history=[];dump(out/'initial.json',initial);print('INITIAL',a.variant,initial['rmse'],flush=True)
 for step in range(1,a.steps+1):
  j=torch.randint(len(fit['x']),(a.batch,),generator=generator)
  lr=3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(step-1)/a.steps))
  for g in opt.param_groups:g['lr']=lr
  opt.zero_grad(set_to_none=True);q=model(fit['x'][j],fit['strip'][j]);target=(fit['y'][j]-fit['base'][j])/20
  m=fit['mask'][j];w=fit['row_weight'][j,None];loss=((q-target).square()*m*w).sum(0)/(m*w).sum(0).clamp_min(1)
  loss.mean().backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step%100==0 or step==a.steps:
   val=evaluate(model,dev);score=float(np.mean(val['rmse']))
   if score<best:
    best=score;best_step=step;torch.save(dict(model=model.state_dict(),config=cfg,step=step),out/'best.pt')
   rec=dict(step=step,rmse=val['rmse'],loss=float(loss.mean()),best_step=best_step,seconds=time.monotonic()-t)
   history.append(rec);dump(out/'history.json',history);dump(out/'progress.json',rec);print(a.variant,rec,flush=True)
   if step-best_step>=300:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);val=evaluate(model,dev)
 dump(out/'result.json',dict(status='complete',best_step=best_step,last_step=step,monitor=val,initial=initial,seconds=time.monotonic()-t))
 print('DONE',a.variant,best_step,val['rmse'],flush=True)
if __name__=='__main__':main()
