"""Point and spatial residual regressors on exactly the same cached inputs."""
import torch
from torch import nn
from torch.nn import functional as F
class ConvBlock(nn.Module):
 def __init__(self,ci,co):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.SiLU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.SiLU())
 def forward(self,x):return self.net(x)
class PointBlock(nn.Module):
 def __init__(self,w):
  super().__init__();self.a=nn.Conv2d(w,w,1);self.b=nn.Conv2d(w,w,1)
 def forward(self,x):return F.silu(x+.2*self.b(F.silu(self.a(x))))
class Model(nn.Module):
 def __init__(self,kind,mean,std):
  super().__init__();self.kind=kind;self.register_buffer('mean',mean.reshape(1,-1,1,1));self.register_buffer('std',std.reshape(1,-1,1,1))
  if kind=='point':
   self.first=nn.Conv2d(47,128,1);self.body=nn.Sequential(*[PointBlock(128) for _ in range(8)]);self.head=nn.Conv2d(128,2,1)
  elif kind=='spatial':
   self.first=nn.Conv2d(47,24,3,padding=1)
   self.enc=nn.ModuleList([ConvBlock(24,24),ConvBlock(24,48),ConvBlock(48,96)])
   self.dec=nn.ModuleList([ConvBlock(96+48,48),ConvBlock(48+24,24)])
   self.head=nn.Conv2d(24,2,1)
  else:raise ValueError(kind)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):
  with torch.autocast('cuda',enabled=False):z=F.silu(self.first(((x.float()-self.mean)/self.std).clamp(-8,8)))
  if self.kind=='point':z=self.body(z.to(torch.bfloat16) if torch.is_autocast_enabled('cuda') else z)
  else:
   skips=[]
   for i,block in enumerate(self.enc):
    if i:z=F.avg_pool2d(z,2)
    z=block(z);skips.append(z)
   for block,skip in zip(self.dec,reversed(skips[:-1])):
    z=block(torch.cat((F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1))
  with torch.autocast('cuda',enabled=False):return self.head(z.float())*5
