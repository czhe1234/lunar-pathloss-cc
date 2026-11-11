"""Native-pixel preserving global operator; mean-block matched control."""
import torch
from torch import nn
from torch.nn import functional as F
from src.c99_axial import Axis,ConvBlock
from src.c117_factorized import terrain_inputs
class FullBlock(nn.Module):
 def __init__(self,c):
  super().__init__();self.row=Axis(c,64,False,True);self.col=Axis(c,64,True,True);self.ff=ConvBlock(c)
 def forward(self,x):return self.ff(self.col(self.row(x)))
def global_inputs(x):
 # Original float32 H/TX derivatives. Never use labels, validity, or residual signs.
 return torch.cat([terrain_inputs(x),x[:,5:7].float(),x[:,7:8].float(),x[:,9:11].float()],1)
class Field(nn.Module):
 def __init__(self,mode):
  super().__init__();assert mode in ['native','mean'];self.mode=mode
  c=384
  self.stem=nn.Conv2d(13*16,c,1)
  self.blocks=nn.ModuleList([FullBlock(c) for _ in range(8)])
  self.film=nn.Linear(58,2*c);nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
  self.unpack=nn.Conv2d(c,96*16,1)
  self.local=nn.Sequential(nn.Conv2d(58,64,1),ConvBlock(64),ConvBlock(64))
  self.mix=nn.Conv2d(160,128,1);self.refine=nn.Sequential(ConvBlock(128),ConvBlock(128))
  self.head=nn.Conv2d(128,2,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def pack(self,x):
  v=global_inputs(x)
  if self.mode=='mean':v=F.avg_pool2d(v,4).repeat_interleave(4,2).repeat_interleave(4,3)
  return F.pixel_unshuffle(v,4)
 def forward(self,x):
  b=len(x);r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  tx=x[torch.arange(b,device=x.device),:,r,c].clamp(-12,12)
  z=self.stem(self.pack(x));ab=self.film(tx).to(z.dtype)[:,:,None,None];a,bias=ab.chunk(2,1);z=z*(1+.1*a.tanh())+.1*bias
  for block in self.blocks:z=block(z)
  z=F.pixel_shuffle(self.unpack(z),4)
  y=self.local(x.clamp(-12,12));z=self.refine(self.mix(torch.cat([z,y],1)))
  return 4*self.head(z)
def create(mode):
 torch.manual_seed(123041);return Field(mode)
