"""C132 shared sparse positive-power transport; calibrated by trainable constitutive laws.
The output is an anchored physical-power ratio, never a free dB residual readout.
"""
import torch
from torch import nn
class Law(nn.Module):
 def __init__(self,out=2):
  super().__init__()
  self.layers=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,out))
  nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,x):
  # FP32 power calculations are outside autocast.
  return 8*torch.tanh(self.layers(x.float()).float()/8)
class PowerTransport(nn.Module):
 def __init__(self,mode='direction',rounds=3):
  super().__init__();self.mode=mode;self.rounds=rounds
  self.source=Law();self.transfer=Law();self.receive=Law();self.measure=Law(4)
 def condition(self,x,kind):
  if self.mode=='direction':return x
  mask=torch.zeros(12,device=x.device);mask[0]=1
  if kind=='source' or kind=='direct':mask[9]=1;mask[11]=1
  return x*mask
 def kernels(self,g):
  with torch.autocast('cuda',torch.bfloat16):
   lf=self.transfer(self.condition(g['kf'],'transfer'))
  k=g['k']*lf.exp()
  mass=torch.zeros(k.shape[0],2048,2,device=k.device).scatter_add_(1,g['src'].long()[...,None].expand(-1,-1,2),k)
  k=k/torch.maximum(torch.ones_like(k),mass.gather(1,g['src'].long()[...,None].expand(-1,-1,2))/.85)
  return k
 def field(self,g,sq,sf,k=None,neutral=False):
  if neutral:q=sq
  else:
   with torch.autocast('cuda',torch.bfloat16):f=self.source(self.condition(sf,'source'))
   q=sq*torch.cat([f,f],1).exp()
  k=g['k'] if neutral else self.kernels(g) if k is None else k
  extra=torch.zeros_like(q);state=q
  src=g['src'].long()[...,None].expand(-1,-1,2);dst=g['dst'].long()[...,None].expand(-1,-1,2)
  for _ in range(self.rounds):
   message=state.gather(1,src)*k
   state=torch.zeros_like(q).scatter_add_(1,dst,message)
   extra=extra+state
  return extra
 def physical(self,g,q,neutral=False,field=None):
  extra=self.field(g,q['sq'],q['sf'],neutral=neutral) if field is None else field
  ri=q['ri'].long();bs=ri.shape[0];index=torch.arange(bs,device=ri.device)[:,None,None]
  area=torch.cat([g['area'],g['area']],1)
  multi=extra[index,ri]/area[index,ri,None].clamp_min(.01)
  if neutral:
   diffuse=((q['rq']+multi)*q['aw'][...,None]).sum(-2)
   direct=q['ds'][...,:2];spec=q['ds'][...,2:]
  else:
   with torch.autocast('cuda',torch.bfloat16):
    qlog=self.source(self.condition(q['qf'],'source'))
    alog=self.receive(self.condition(q['af'],'receiver'))
    dlog=self.measure(self.condition(q['df'],'direct'))
   diffuse=((q['rq']*qlog.exp()+multi)*(q['aw'][...,None]*alog.exp())).sum(-2)
   direct=q['ds'][...,:2]*dlog[...,:2].exp();spec=q['ds'][...,2:]*dlog[...,2:].exp()
  return (direct+spec+diffuse).clamp_min(1e-30)
 def forward(self,g,q,reference):
  value=self.physical(g,q)
  g0=q['g0'] if 'g0' in q else self.physical(g,q,neutral=True).detach()
  return reference-10/2.302585092994046*(value.log()-g0.log())
