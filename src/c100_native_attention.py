"""C100 retain1m remote terrain before strided compression; three matched warm controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.c99_axial import AxialField,Axis,ConvBlock
class NativeAxis(Axis):
 def __init__(self,vertical,global_context):
  super().__init__(48,256,vertical,global_context)
  self.heads=3;self.dim=16
  phase=torch.arange(256)[:,None]*torch.exp(-math.log(10000.)*torch.arange(0,self.dim,2)[None]/self.dim)
  self.cos=phase.cos()[None,None];self.sin=phase.sin()[None,None]
  nn.init.zeros_(self.scale)
 def forward(self,x):
  if self.vertical:x=x.transpose(2,3)
  b,c,h,w=x.shape;assert w==self.length
  line=x.permute(0,2,3,1).reshape(b*h,w,c)
  q,k,v=self.qkv(self.norm(line)).reshape(b*h,w,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
  q=self.rope(q);k=self.rope(k)
  if not self.global_context:
   def cut(a):return a.reshape(b*h,self.heads,w//16,16,self.dim).permute(0,2,1,3,4).reshape(-1,self.heads,16,self.dim)
   q,k,v=[cut(a) for a in [q,k,v]]
  # Independent window batches can exceed CUDA launch grid limits at native resolution.
  # Splitting this batch axis changes neither neighborhood nor attention formula.
  if q.shape[0]>32768:
   value=torch.cat([F.scaled_dot_product_attention(q[j:j+32768],k[j:j+32768],v[j:j+32768],dropout_p=0.) for j in range(0,len(q),32768)],0)
  else:value=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
  if not self.global_context:value=value.reshape(b*h,w//16,self.heads,16,self.dim).permute(0,2,1,3,4).reshape(b*h,self.heads,w,self.dim)
  value=value.transpose(1,2).reshape(b*h,w,c);value=self.out(value)*self.scale.to(value.dtype)
  y=x+value.reshape(b,h,w,c).permute(0,3,1,2)
  return y.transpose(2,3) if self.vertical else y
class NativeBlock(nn.Module):
 def __init__(self,g):
  super().__init__();self.row=NativeAxis(False,g);self.col=NativeAxis(True,g);self.conv=ConvBlock(48)
  nn.init.zeros_(self.conv.scale)
 def forward(self,x):return self.conv(self.col(self.row(x)))
class NativeField(AxialField):
 def __init__(self,mode):
  super().__init__(True);assert mode in ['continue','local','global'];self.mode=mode
  self.extra=NativeBlock(mode=='global')
 def forward(self,x):
  b=x.shape[0];r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  x=x.clamp(-12,12);tx=x[torch.arange(b,device=x.device),:,r,c]
  fine=self.native(self.stem(x))
  if self.mode!='continue':fine=self.extra(fine)
  middle=self.stage1(self.condition(self.down1(fine),tx,self.film1))
  coarse=self.stage2(self.condition(self.down2(middle),tx,self.film2))
  mid=self.decode1(middle+F.interpolate(self.up1(coarse),size=middle.shape[-2:],mode='bilinear',align_corners=False))
  y=self.decode0(fine+F.interpolate(self.up0(mid),size=fine.shape[-2:],mode='bilinear',align_corners=False))
  return 4*self.head(y)
def initialize(mode,parent):
 torch.manual_seed(100041);net=NativeField(mode);missing,unexpected=net.load_state_dict(parent,strict=False)
 assert not unexpected and all(k.startswith('extra.') for k in missing)
 return net
