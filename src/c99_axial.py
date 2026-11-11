"""Independent TX-conditioned axial field network; local versus global context only."""
import math
import torch
from torch import nn
from torch.nn import functional as F
class ConvBlock(nn.Module):
 def __init__(self,c):
  super().__init__();self.dw=nn.Conv2d(c,c,5,padding=2,groups=c);self.norm=nn.LayerNorm(c)
  self.up=nn.Linear(c,2*c);self.down=nn.Linear(2*c,c);self.scale=nn.Parameter(torch.full((c,),.1))
 def forward(self,x):
  y=self.dw(x).permute(0,2,3,1);y=self.down(F.gelu(self.up(self.norm(y))))
  return x+(y*self.scale.to(y.dtype)).permute(0,3,1,2)
class Axis(nn.Module):
 def __init__(self,c,length,vertical,global_context):
  super().__init__();self.c=c;self.heads=4;self.dim=c//4;self.length=length;self.vertical=vertical;self.global_context=global_context
  assert self.dim%2==0 and length%16==0
  self.norm=nn.LayerNorm(c);self.qkv=nn.Linear(c,3*c);self.out=nn.Linear(c,c)
  self.scale=nn.Parameter(torch.full((c,),.1))
  phase=torch.arange(length)[:,None]*torch.exp(-math.log(10000.)*torch.arange(0,self.dim,2)[None]/self.dim)
  self.register_buffer('cos',phase.cos()[None,None],persistent=False);self.register_buffer('sin',phase.sin()[None,None],persistent=False)
 def rope(self,q):
  dtype=q.dtype;a,b=q.float().reshape(*q.shape[:-1],self.dim//2,2).unbind(-1)
  return torch.stack([a*self.cos-b*self.sin,b*self.cos+a*self.sin],-1).flatten(-2).to(dtype)
 def forward(self,x):
  if self.vertical:x=x.transpose(2,3)
  b,c,h,w=x.shape;assert w==self.length
  line=x.permute(0,2,3,1).reshape(b*h,w,c)
  q,k,v=self.qkv(self.norm(line)).reshape(b*h,w,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
  q=self.rope(q);k=self.rope(k)
  if not self.global_context:
   def cut(a):return a.reshape(b*h,self.heads,w//16,16,self.dim).permute(0,2,1,3,4).reshape(-1,self.heads,16,self.dim)
   q,k,v=[cut(a) for a in [q,k,v]]
  value=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
  if not self.global_context:value=value.reshape(b*h,w//16,self.heads,16,self.dim).permute(0,2,1,3,4).reshape(b*h,self.heads,w,self.dim)
  value=value.transpose(1,2).reshape(b*h,w,c);value=self.out(value)*self.scale.to(value.dtype)
  y=x+value.reshape(b,h,w,c).permute(0,3,1,2)
  return y.transpose(2,3) if self.vertical else y
class AxialBlock(nn.Module):
 def __init__(self,c,n,g):
  super().__init__();self.row=Axis(c,n,False,g);self.col=Axis(c,n,True,g);self.conv=ConvBlock(c)
 def forward(self,x):return self.conv(self.col(self.row(x)))
class AxialField(nn.Module):
 def __init__(self,global_context,channels=56):
  super().__init__();self.global_context=global_context
  self.stem=nn.Conv2d(channels,48,1);self.native=nn.Sequential(ConvBlock(48),ConvBlock(48))
  self.down1=nn.Conv2d(48,96,3,stride=2,padding=1);self.stage1=nn.Sequential(AxialBlock(96,128,global_context),AxialBlock(96,128,global_context))
  self.down2=nn.Conv2d(96,160,3,stride=2,padding=1);self.stage2=nn.Sequential(AxialBlock(160,64,global_context),AxialBlock(160,64,global_context))
  self.film1=nn.Linear(channels,192);self.film2=nn.Linear(channels,320)
  for m in [self.film1,self.film2]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
  self.up1=nn.Conv2d(160,96,1);self.decode1=nn.Sequential(ConvBlock(96),ConvBlock(96))
  self.up0=nn.Conv2d(96,48,1);self.decode0=nn.Sequential(ConvBlock(48),ConvBlock(48))
  self.head=nn.Conv2d(48,2,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def condition(self,x,v,linear):
  ab=linear(v).to(x.dtype)[:,:,None,None];a,b=ab.chunk(2,1)
  return x*(1+.1*a.tanh())+.1*b
 def forward(self,x):
  b=x.shape[0];r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  x=x.clamp(-12,12);tx=x[torch.arange(b,device=x.device),:,r,c]
  fine=self.native(self.stem(x))
  middle=self.stage1(self.condition(self.down1(fine),tx,self.film1))
  coarse=self.stage2(self.condition(self.down2(middle),tx,self.film2))
  mid=self.decode1(middle+F.interpolate(self.up1(coarse),size=middle.shape[-2:],mode='bilinear',align_corners=False))
  y=self.decode0(fine+F.interpolate(self.up0(mid),size=fine.shape[-2:],mode='bilinear',align_corners=False))
  return 4*self.head(y)
