"""Full-terrain signed residual state and conditional mean operator."""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from src.c123_lossless import Field
class StateField(Field):
 def __init__(self,prior,means):
  super().__init__('native');self.head=nn.Conv2d(128,12,1)
  self.register_buffer('prior',prior.float());self.register_buffer('means',means.float())
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
  with torch.no_grad():
   b=self.head.bias.reshape(2,6);mu0=-(prior[:,1]*means[:,1]+prior[:,2]*means[:,2])/prior[:,0]
   assert (mu0.abs()<5).all() and (means[:,1]>5).all() and (means[:,2]<-5).all()
   b[:,3]=torch.atanh(mu0/5);b[:,4]=torch.log(torch.expm1(means[:,1]-5));b[:,5]=torch.log(torch.expm1(-means[:,2]-5))
 def outputs(self,x):
  n=len(x);r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  tx=x[torch.arange(n,device=x.device),:,r,c].clamp(-12,12);z=self.stem(self.pack(x));ab=self.film(tx).to(z.dtype)[:,:,None,None];scale,bias=ab.chunk(2,1);z=z*(1+.1*scale.tanh())+.1*bias
  for block in self.blocks:z=block(z)
  z=F.pixel_shuffle(self.unpack(z),4);local=self.local(x.clamp(-12,12));z=self.refine(self.mix(torch.cat([z,local],1)))
  with torch.autocast('cuda',enabled=False):raw=self.head(z.float()).reshape(n,2,6,256,256)
  logits=raw[:,:,:3];mu=torch.stack([5*torch.tanh(raw[:,:,3]),5+F.softplus(raw[:,:,4]),-5-F.softplus(raw[:,:,5])],2)
  p=(logits+self.prior.log()[None,:,:,None,None]).softmax(2);delta=(p*mu).sum(2)
  return delta,logits,mu,p
 def forward(self,x):return self.outputs(x)[0]
def losses(delta,logits,mu,base,y,valid,weight,prior,mode):
 r=y-base;cls=torch.where(r>=5,1,torch.where(r<=-5,2,0)).long()
 w=valid*weight[:,None,None,None];den=w.sum((0,2,3)).clamp_min(1)
 e=torch.where(valid,delta-r,0.);s=(e.square()*weight[:,None,None,None]).sum((0,2,3));risk=(s/den).mean()/25
 classprior=prior[torch.arange(2,device=r.device)[None,:,None,None],cls]
 ce=F.cross_entropy(logits.reshape(-1,3,256,256),cls.reshape(-1,256,256),reduction='none').reshape_as(r)
 pred=mu.gather(2,cls[:,:,None]).squeeze(2);amp=torch.where(valid,(pred-r).square(),0.)
 balanced=w/(3*classprior.clamp_min(1e-7));closs=((ce*balanced).sum((0,2,3))/den).mean();aloss=((amp*balanced).sum((0,2,3))/den).mean()/25
 total=risk+(.25*(closs+aloss) if mode=='state' else 0.)
 return total,dict(risk=risk,classCE=closs,amplitude=aloss),s,den
@torch.inference_mode()
def evaluate(net,data,batch=8,force_base=False,save=None):
 arr=np.lib.format.open_memmap(save,'w+',dtype=np.float32,shape=(len(data.ids),2,256,256)) if save is not None else None
 net.eval();s=torch.zeros(2,device='cuda',dtype=torch.float64);count=s.clone();tail_s=s.clone();tail_n=s.clone();oracle_s=s.clone();gate=s.clone();conf=torch.zeros(2,3,3,device='cuda',dtype=torch.float64)
 for st in range(0,len(data.ids),batch):
  x,b,y,v=data.batch(slice(st,min(st+batch,len(data.ids))))
  with torch.autocast('cuda',torch.bfloat16):d,logits,mu,p=net.outputs(x)
  if force_base:d=torch.zeros_like(d)
  if arr is not None:arr[st:min(st+batch,len(data.ids))]=(b+d).float().cpu().numpy()
  r=y-b;cls=torch.where(r>=5,1,torch.where(r<=-5,2,0)).long();e=torch.where(v,d-r,0.);s+=e.double().square().sum((0,2,3));count+=v.sum((0,2,3))
  tail=v&(cls!=0);tail_s+=torch.where(tail,e.double().square(),0).sum((0,2,3));tail_n+=tail.sum((0,2,3));gate+=(torch.where(tail,p[:,:,1]+p[:,:,2],0)).sum((0,2,3),dtype=torch.float64)
  true_mu=mu.gather(2,cls[:,:,None]).squeeze(2);oracle_s+=torch.where(v,(true_mu-r).double().square(),0).sum((0,2,3))
  pred=p.argmax(2)
  for bb in range(2):
   code=cls[:,bb][v[:,bb]]*3+pred[:,bb][v[:,bb]];conf[bb]+=torch.bincount(code,minlength=9).reshape(3,3)
 if arr is not None:arr.flush()
 rmse=(s/count).sqrt().tolist();net.train()
 return dict(rmse=rmse,sse=s.tolist(),count=count.tolist(),tail_rmse=(tail_s/tail_n).sqrt().tolist(),tail_count=tail_n.tolist(),true_class_oracle_rmse=(oracle_s/count).sqrt().tolist(),mean_tail_probability_on_true_tail=(gate/tail_n).tolist(),state_confusion=conf.tolist())
