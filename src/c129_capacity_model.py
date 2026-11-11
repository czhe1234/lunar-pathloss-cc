"""Width-only capacity family with function-preserving pretrained expansion."""
import math,torch
from torch import nn
from src.c99_axial import Axis,ConvBlock,AxialField
from src.c127_joint import JointField,TxField,SceneField
from src.c127_train import create
class WideAxis(Axis):
 def __init__(self,c,length,vertical,factor):
  nn.Module.__init__(self);self.c=c;self.heads=4*factor;self.dim=c//self.heads;self.length=length;self.vertical=vertical;self.global_context=True
  self.norm=nn.LayerNorm(c);self.qkv=nn.Linear(c,3*c);self.out=nn.Linear(c,c);self.scale=nn.Parameter(torch.full((c,),.1))
  phase=torch.arange(length)[:,None]*torch.exp(-math.log(10000.)*torch.arange(0,self.dim,2)[None]/self.dim)
  self.register_buffer('cos',phase.cos()[None,None],persistent=False);self.register_buffer('sin',phase.sin()[None,None],persistent=False)
class WideBlock(nn.Module):
 def __init__(self,c,n,k):
  super().__init__();self.row=WideAxis(c,n,False,k);self.col=WideAxis(c,n,True,k);self.conv=ConvBlock(c)
 def forward(self,x):return self.conv(self.col(self.row(x)))
class WideTx(nn.Module):
 condition=AxialField.condition
 forward=TxField.forward
 def __init__(self,k):
  super().__init__();a,b,c=[v*k for v in [48,96,160]]
  self.stem=nn.Conv2d(56,a,1);self.native=nn.Sequential(ConvBlock(a),ConvBlock(a))
  self.down1=nn.Conv2d(a,b,3,stride=2,padding=1);self.stage1=nn.Sequential(WideBlock(b,128,k),WideBlock(b,128,k))
  self.down2=nn.Conv2d(b,c,3,stride=2,padding=1);self.stage2=nn.Sequential(WideBlock(c,64,k),WideBlock(c,64,k))
  self.film1=nn.Linear(56,2*b);self.film2=nn.Linear(56,2*c)
  self.up1=nn.Conv2d(c,b,1);self.decode1=nn.Sequential(ConvBlock(b),ConvBlock(b))
  self.up0=nn.Conv2d(b,a,1);self.decode0=nn.Sequential(ConvBlock(a),ConvBlock(a))
  self.head=nn.Conv2d(a,2,1)
class WideField(nn.Module):
 components=JointField.components
 forward=JointField.forward
 def __init__(self,k):
  super().__init__();self.tx=WideTx(k);self.scene=SceneField()
def expand_state(old,new):
 target=new.state_dict();source=old.state_dict()
 for name,v in target.items():
  u=source[name]
  if u.shape==v.shape:v.copy_(u);continue
  ratio=[a//b for a,b in zip(v.shape,u.shape)]
  assert all(a==b*k for a,b,k in zip(v.shape,u.shape,ratio)),(name,u.shape,v.shape)
  row=torch.arange(u.shape[0])
  parts=3 if '.qkv.' in name else (2 if name.startswith('tx.film') else 1)
  if ratio[0]>1:
   row=row.reshape(parts,-1).repeat(1,ratio[0]).flatten()
  value=u.index_select(0,row)
  if u.ndim>=2 and ratio[1]>1:
   value=value.index_select(1,torch.arange(u.shape[1]).repeat(ratio[1]))/ratio[1]
  assert value.shape==v.shape,(name,value.shape,v.shape)
  v.copy_(value)
 new.load_state_dict(target)
def initialize(k,jitter=True):
 torch.manual_seed(129041);old=create('residual')
 if k==1:net=old
 else:
  net=WideField(k);expand_state(old,net)
 if jitter:
  gen=torch.Generator().manual_seed(129043)
  with torch.no_grad():
   for name,p in net.named_parameters():
    if p.ndim>=2 and 'head' not in name:
     # Same relative perturbation in both widths; breaks duplicated-channel symmetry.
     p.add_(torch.randn(p.shape,generator=gen,dtype=p.dtype)*p.std()*.001)
 return net
