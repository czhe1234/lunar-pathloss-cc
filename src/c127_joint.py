"""Input-only scene + TX field, with optional grouped component supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from src.c99_axial import AxialField,ConvBlock,AxialBlock
from src.c117_factorized import terrain_inputs

class TxField(AxialField):
 def forward(self,x):
  n=x.shape[0];r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  x=x.clamp(-12,12);tx=x[torch.arange(n,device=x.device),:,r,c]
  fine=self.native(self.stem(x))
  middle=self.stage1(self.condition(self.down1(fine),tx,self.film1))
  coarse=self.stage2(self.condition(self.down2(middle),tx,self.film2))
  mid=self.decode1(middle+F.interpolate(self.up1(coarse),size=middle.shape[-2:],mode='bilinear',align_corners=False))
  y=self.decode0(fine+F.interpolate(self.up0(mid),size=fine.shape[-2:],mode='bilinear',align_corners=False))
  with torch.autocast('cuda',enabled=False):return 4*self.head(y.float())

class SceneField(nn.Module):
 def __init__(self):
  super().__init__()
  self.stem=nn.Conv2d(8,24,1);self.fine=ConvBlock(24)
  self.down1=nn.Conv2d(24,48,3,stride=2,padding=1);self.mid=ConvBlock(48)
  self.down2=nn.Conv2d(48,96,3,stride=2,padding=1)
  self.global_blocks=nn.Sequential(AxialBlock(96,64,True),AxialBlock(96,64,True))
  self.up1=nn.Conv2d(96,48,1);self.refine1=ConvBlock(48)
  self.up0=nn.Conv2d(48,24,1);self.refine0=ConvBlock(24)
  self.head=nn.Conv2d(24,2,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):
  with torch.autocast('cuda',enabled=False):z=self.stem(terrain_inputs(x))
  z=self.fine(z);m=self.mid(self.down1(z));g=self.global_blocks(self.down2(m))
  m=self.refine1(m+F.interpolate(self.up1(g),size=m.shape[-2:],mode='bilinear',align_corners=False))
  z=self.refine0(z+F.interpolate(self.up0(m),size=z.shape[-2:],mode='bilinear',align_corners=False))
  with torch.autocast('cuda',enabled=False):return 4*self.head(z.float())

class JointField(nn.Module):
 def __init__(self):
  super().__init__();self.tx=TxField(True,channels=56);self.scene=SceneField()
 def components(self,x,first=None,inverse=None):
  # first/inverse only deduplicate identical terrain inputs; not model features.
  s=self.scene(x if first is None else x[first])
  if inverse is not None:s=s[inverse]
  return s,self.tx(x)
 def forward(self,x,first=None,inverse=None):
  s,t=self.components(x,first,inverse);return s+t

def objective(s,t,reference,y,valid,weight,group,mode):
 p=reference+s+t
 error=torch.where(valid,p-y,0.)
 w=valid*weight[:,None,None,None];count=w.sum((0,2,3)).clamp_min(1)
 ss=(error.square()*weight[:,None,None,None]).sum((0,2,3))
 risk=(ss/count).mean()/25
 scene_ss=torch.zeros(2,device=y.device);scene_n=scene_ss.clone();tx_ss=scene_ss.clone();tx_n=scene_ss.clone()
 for members in group:
  if len(members)<2:continue
  ix=torch.as_tensor(members,device=y.device);v=valid[ix];num=v.sum(0);ok=num>=2
  residual=torch.where(v,y[ix]-reference[ix],0.)
  mean=residual.sum(0)/num.clamp_min(1)
  # Same scene output for all TX, supervised once per terrain/band/pixel.
  sw=weight[ix].mean()*ok
  scene_ss+=(torch.where(ok,s[ix[0]]-mean,0.).square()*sw).sum((1,2));scene_n+=sw.sum((1,2))
  tv=v&ok[None];tw=tv*weight[ix,None,None,None]
  te=torch.where(tv,t[ix]-(y[ix]-reference[ix]-mean[None]),0.)
  tx_ss+=(te.square()*weight[ix,None,None,None]).sum((0,2,3));tx_n+=tw.sum((0,2,3))
 scene_loss=(scene_ss/scene_n.clamp_min(1)).mean()/25
 tx_loss=(tx_ss/tx_n.clamp_min(1)).mean()/25
 common=valid[:,0]&valid[:,1];bw=common*weight[:,None,None]
 be=torch.where(common,(p[:,1]-p[:,0])-(y[:,1]-y[:,0]),0.)
 band_loss=(be.square()*weight[:,None,None]).sum()/bw.sum().clamp_min(1)/50
 loss=risk+(.25*.5*(scene_loss+tx_loss)+.25*band_loss if mode=='joint' else 0.)
 return loss,dict(risk=risk,scene=scene_loss,tx=tx_loss,band=band_loss),ss,count
