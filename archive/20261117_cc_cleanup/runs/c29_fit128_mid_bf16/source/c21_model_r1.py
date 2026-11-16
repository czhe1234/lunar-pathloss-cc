"""C21 revision1: retain spatial contrast before normalizing scene tokens.
The original pre-norm scene blocks learned a shared token offset much larger
than their spatial differences. Centering over terrain tokens removes that
shared offset before channel LayerNorm. All other architecture is unchanged.
"""
from torch import nn
from c21_model import Model as OriginalModel
class SpatialCenterNorm(nn.LayerNorm):
 def forward(self,x):
  return super().forward(x-x.mean(dim=1,keepdim=True))
class Model(OriginalModel):
 def __init__(self,mean,std,geometry=True,width=256,layers=4):
  super().__init__(mean,std,geometry,width,layers)
  self.knorm=SpatialCenterNorm(width)
