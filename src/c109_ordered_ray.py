"""Same-parameter ordered/permuted path-reader; raw H/TX information is unchanged."""
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from src.c105_context import create as parent
class OrderedRay(nn.Module):
 def __init__(self,mode):
  super().__init__();assert mode in ['ordered','permuted'];self.mode=mode
  self.trunk=parent();self.trunk.head=nn.Identity();torch.manual_seed(109041)
  self.path=nn.Sequential(nn.Conv3d(8,16,1),nn.SiLU(),nn.Conv3d(16,16,(3,1,1),padding=(1,0,0),groups=16),nn.SiLU(),
   nn.Conv3d(16,16,1),nn.SiLU(),nn.Conv3d(16,16,(3,1,1),padding=(1,0,0),groups=16),nn.SiLU())
  self.readout=nn.Sequential(nn.Conv2d(80,64,1),nn.SiLU(),nn.Conv2d(64,2,1))
  nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
  self.register_buffer('fraction',(torch.arange(32,dtype=torch.float32)+.5)/32)
  self.register_buffer('permutation',torch.tensor(np.random.default_rng(109017).permutation(32),dtype=torch.int64))
 @torch.no_grad()
 def profile(self,x):
  with torch.autocast('cuda',enabled=False):
   x=x.float();b,_,hh,ww=x.shape;assert hh==ww==256
   h=x[:,0]*100;dx=x[:,5]*128;dy=x[:,6]*128
   r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
   # Keep the existing basic_inputs height convention; no geometry-offset reconstruction.
   z=h[torch.arange(b,device=x.device),r,c]+3;dz=h+1-z[:,None,None]
   distance=(dx.square()+dy.square()+dz.square()).sqrt().clamp_min(.25);rh=(dx.square()+dy.square()).sqrt().clamp_min(1.)
   t=self.fraction[None,:,None,None]
   gx=(c[:,None,None,None]+dx[:,None]*t)/127.5-1;gy=(r[:,None,None,None]+dy[:,None]*t)/127.5-1
   grid=torch.stack((gx,gy),-1).reshape(b,32*256,256,2)
   fields=torch.cat((h[:,None],x[:,3:5],x[:,56:58]),1)
   s=F.grid_sample(fields,grid,mode='bilinear',padding_mode='border',align_corners=True).reshape(b,5,32,256,256)
   clear=s[:,0]-(z[:,None,None,None]+dz[:,None]*t)
   fresnel=distance[:,None]*t*(1-t)
   slope=5*(s[:,1]*dx[:,None]+s[:,2]*dy[:,None])/rh[:,None]
   values=[torch.asinh(clear/2)/3,torch.asinh(clear/(fresnel*(299792458/415e6)).sqrt().clamp_min(.02))/3,
    torch.asinh(clear/(fresnel*(299792458/5.775e9)).sqrt().clamp_min(.02))/3,torch.asinh(slope)/3,
    s[:,3],s[:,4],(2*t-1).expand(b,-1,256,256),x[:,8,None].expand(-1,32,-1,-1)]
   p=torch.stack(values,1).clamp(-12,12)
   return p if self.mode=='ordered' else p.index_select(2,self.permutation)
 def forward(self,x):
  context=self.trunk(x)/4
  p=self.path(self.profile(x));ray=torch.cat((p.mean(2),p.amax(2)),1)
  return 4*self.readout(torch.cat((context,ray.to(context.dtype)),1))
def create(mode):return OrderedRay(mode)
