"""Explicit floor-height-corrected ordered ray and receiver height phases.
These are input transforms, not recovered simulator coordinates or extra observations.
"""
import math
import torch
from src.c34_model import Model as Base
def expand_features(x):
 with torch.autocast('cuda',enabled=False):
  x=x.float();t=(torch.arange(128,device=x.device)+.5)/128
  frac=torch.remainder(x[:,31],1)
  clearance=2*torch.sinh(3*x[:,104:232])+frac[:,None]*(1-t)
  profile=torch.asinh(clearance/2)/3
  rxh=x[:,0]*100
  phase=2*math.pi*rxh[:,None]/torch.tensor([.25,.5,1.,2.],device=x.device)
  return torch.cat((x,profile,phase.sin(),phase.cos()),1)
class Model(Base):
 def forward(self,x):
  return super().forward(expand_features(x))
