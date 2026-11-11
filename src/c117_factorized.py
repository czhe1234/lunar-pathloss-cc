"""Terrain-only spatial representation and late source/receiver interaction."""
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Module):
 def __init__(self,c):
  super().__init__();self.dw=nn.Conv2d(c,c,5,padding=2,groups=c)
  self.norm=nn.LayerNorm(c);self.up=nn.Linear(c,2*c);self.down=nn.Linear(2*c,c)
  self.scale=nn.Parameter(torch.full((c,),.1))
 def forward(self,x):
  z=self.dw(x).permute(0,2,3,1);z=self.down(F.gelu(self.up(self.norm(z))))
  return x+(z*self.scale.to(z.dtype)).permute(0,3,1,2)
def terrain_inputs(x):
 h=x[:,0:1].float()*100
 sx=5*x[:,3:4].float();sy=5*x[:,4:5].float()
 def hp(k):return h-F.avg_pool2d(F.pad(h,(k//2,)*4,mode='replicate'),k,stride=1)
 mean=h.mean((2,3),keepdim=True)
 return torch.cat([h/100,(h-mean)/10,sx/.25,sy/.25,hp(3)/.05,hp(7)/.2,hp(15)/.5,hp(31)/2],1).clamp(-15,15)
class Field(nn.Module):
 def __init__(self,mode):
  super().__init__();assert mode in ['early','factorized'];self.mode=mode
  c=[48,80,128,192,256]
  self.terrain_stem=nn.Conv2d(8,c[0],1);self.condition_stem=nn.Conv2d(58,c[0],1,bias=False)
  self.stages=nn.ModuleList([nn.Sequential(Block(k),Block(k)) for k in c])
  self.down=nn.ModuleList([nn.Conv2d(c[j],c[j+1],3,stride=2,padding=1) for j in range(4)])
  self.up=nn.ModuleList([nn.Conv2d(c[j+1],c[j],1) for j in range(4)])
  self.decode=nn.ModuleList([Block(c[j]) for j in range(4)])
  self.source=nn.Linear(48,48);self.receiver=nn.Conv2d(48,48,1)
  self.query=nn.Conv2d(58,48,1)
  self.mix=nn.Conv2d(48*4,96,1)
  self.readout=nn.Sequential(*[nn.Sequential(nn.Conv2d(96,96,1),nn.GELU()) for _ in range(3)])
  self.head=nn.Conv2d(96,2,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def encode(self,x):
  z=self.terrain_stem(terrain_inputs(x))
  if self.mode=='early':z=z+self.condition_stem(x.clamp(-12,12))
  z=self.stages[0](z);skip=[z]
  for j in range(4):z=self.stages[j+1](self.down[j](z));skip.append(z)
  for j in range(3,-1,-1):z=self.decode[j](skip[j]+F.interpolate(self.up[j](z),size=skip[j].shape[-2:],mode='bilinear',align_corners=False))
  return z
 def forward(self,x):
  b=len(x);r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  z=self.encode(x)
  source=self.source(z[torch.arange(b,device=x.device),:,r,c])[:,:,None,None]
  rx=self.receiver(z);q=self.query(x.clamp(-12,12))
  interaction=rx*torch.tanh(source)
  v=self.mix(torch.cat([z,source.expand(-1,-1,256,256),q,interaction],1))
  v=v+self.readout(v);return 4*self.head(v)
def create(mode):
 torch.manual_seed(117041);return Field(mode)
