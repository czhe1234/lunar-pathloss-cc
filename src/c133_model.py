"""Terrain-conditioned constitutive kernels at actual propagation endpoints.
K uses terrain alone. TX enters source illumination and direct measurement conditions.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.c132_model import PowerTransport
class Block(nn.Module):
 def __init__(self,c,d=1):
  super().__init__()
  self.net=nn.Sequential(nn.GroupNorm(4,c),nn.Conv2d(c,c,3,padding=d,dilation=d,groups=c),nn.Conv2d(c,c*2,1),nn.SiLU(),nn.Conv2d(c*2,c,1))
 def forward(self,x):return x+self.net(x)
class TerrainContext(nn.Module):
 def __init__(self):
  super().__init__()
  self.stem=nn.Conv2d(5,32,3,padding=1);self.fine=Block(32)
  self.down=nn.Conv2d(32,48,4,stride=4);self.globalpart=nn.Sequential(Block(48,1),Block(48,4),Block(48,12))
  self.up=nn.Conv2d(48,32,1);self.out=nn.Sequential(Block(32),nn.Conv2d(32,16,1))
 def forward(self,h):
  # No TX, label, mask, or terrain identifier enters this encoder.
  h=h[:,None];sx=(F.pad(h,(1,1,0,0),mode='replicate')[:,:,:,2:]-F.pad(h,(1,1,0,0),mode='replicate')[:,:,:,:-2])*.5
  sy=(F.pad(h,(0,0,1,1),mode='replicate')[:,:,2:]-F.pad(h,(0,0,1,1),mode='replicate')[:,:,:-2])*.5
  local=F.avg_pool2d(F.pad(h,(4,4,4,4),mode='replicate'),9,stride=1)
  inp=torch.cat([h/128,(h-local)/8,sx/4,sy/4,torch.sqrt(sx.square()+sy.square()+1e-6)/8],1).clamp(-8,8)
  f=self.fine(self.stem(inp));m=self.globalpart(self.down(f));return self.out(f+F.interpolate(self.up(m),size=(256,256),mode='bilinear',align_corners=False))
class Law(nn.Module):
 def __init__(self,out):
  super().__init__();self.layers=nn.Sequential(nn.Linear(44,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,out))
  nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,x):return 8*torch.tanh(self.layers(x).float()/8)
class ContextTransport(PowerTransport):
 def __init__(self,mode='direction',rounds=3):
  super().__init__(mode,rounds);self.encoder=TerrainContext();self.source=Law(2);self.transfer=Law(2);self.receive=Law(2);self.measure=Law(4)
  j=torch.arange(1024);c=torch.floor((j%32+.5)*255/32).clamp_max(254);r=torch.floor((j//32+.5)*255/32).clamp_max(254)
  self.register_buffer('node_grid',torch.stack([(c+2/3)/127.5-1,(r+1/3)/127.5-1],1)[None])
 def sample(self,x,xy):
  b=xy.shape[0];shape=xy.shape[1:-1];y=F.grid_sample(x.float(),xy.float().reshape(b,1,-1,2),mode='bilinear',align_corners=True)
  return y[:,:,0].transpose(1,2).reshape(b,*shape,16)
 def forward(self,g,q,reference):
  with torch.autocast('cuda',torch.bfloat16):feat=self.encoder(q['h'])
  feat=feat[q['inverse']]
  b=len(reference);node=self.sample(feat,self.node_grid.expand(b,-1,-1))
  rx=self.sample(feat,q['rxxy']);hit=self.sample(feat,q['hitxy']);tx=self.sample(feat,q['txxy'][:,None])[:,0]
  bi=torch.arange(b,device=feat.device)[:,None]
  src=g['src'].long();dst=g['dst'].long()
  kg=torch.cat([self.condition(g['kf'].float(),'transfer'),node[bi,src%1024],node[bi,dst%1024]],-1)
  sf=torch.cat([self.condition(q['sf'].float(),'source'),node,tx[:,None].expand(-1,1024,-1)],-1)
  with torch.autocast('cuda',torch.bfloat16):
   kl=self.transfer(kg);sl=self.source(sf)
  k=g['k']*kl.exp();mass=torch.zeros(b,2048,2,device=feat.device).scatter_add_(1,src[...,None].expand(-1,-1,2),k)
  k=k/torch.maximum(torch.ones_like(k),mass.gather(1,src[...,None].expand(-1,-1,2))/.85)
  self.last_k=k.detach()
  state=q['sq']*torch.cat([sl,sl],1).exp();extra=torch.zeros_like(state)
  for _ in range(self.rounds):
   state=torch.zeros_like(state).scatter_add_(1,dst[...,None].expand(-1,-1,2),state.gather(1,src[...,None].expand(-1,-1,2))*k)
   extra=extra+state
  ri=q['ri'].long();idx=torch.arange(b,device=feat.device)[:,None,None];area=torch.cat([g['area'],g['area']],1)
  multi=extra[idx,ri]/area[idx,ri,None].clamp_min(.01)
  npix=q['rq'].shape[1]
  qi=torch.cat([self.condition(q['qf'].float(),'source'),hit,tx[:,None,None].expand(-1,npix,48,-1)],-1)
  ai=torch.cat([self.condition(q['af'].float(),'receiver'),hit,rx[:,:,None].expand(-1,-1,48,-1)],-1)
  di=torch.cat([self.condition(q['df'].float(),'direct'),rx,tx[:,None].expand(-1,npix,-1)],-1)
  with torch.autocast('cuda',torch.bfloat16):
   qlog=self.source(qi);alog=self.receive(ai);dlog=self.measure(di)
  diff=((q['rq']*qlog.exp()+multi)*q['aw'][...,None]*alog.exp()).sum(-2)
  gain=(q['ds'][...,:2]*dlog[...,:2].exp()+q['ds'][...,2:]*dlog[...,2:].exp()+diff).clamp_min(1e-30)
  return reference-10/2.302585092994046*(gain.log()-q['g0'].log())
