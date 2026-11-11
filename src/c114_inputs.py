"""Two existing H/TX height conventions, same58raw-derived input channels."""
import torch
from src.c105_context import ResidualPanel as Parent
class InputPanel(Parent):
 geometry_mode='old'
 def batch(self,ix):
  x,base,y,w=super().batch(ix)
  if self.geometry_mode=='old':return x,base,y,w
  h=self.heights[self.hidx[ix]];r=self.tr[ix];c=self.tc[ix];b=len(h)
  z=h[torch.arange(b,device=h.device),(r+1).clamp_max(255),c].floor()+3
  dz=h+1-z[:,None,None];dx=x[:,5]*128;dy=x[:,6]*128
  distance=(dx.square()+dy.square()+dz.square()).sqrt().clamp_min(.25)
  x[:,1]=(h-(z-3)[:,None,None])/50;x[:,7]=dz/50;x[:,8]=torch.log1p(distance)/6
  freq=torch.tensor([415e6,5.775e9],device=h.device)[None,:,None,None]
  fs=20*torch.log10(distance[:,None]*freq*(4*torch.pi/299792458.))
  x[:,56:58]=(base-fs)/20
  return x,base,y,w
