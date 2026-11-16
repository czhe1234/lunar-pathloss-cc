"""C152: C82-preserving raw68 Field with aligned all-physical Global70 (raw68 input)."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from src.c123_lossless import Field as OriginalField, global_inputs
from src.c150_field import normalize_diffraction
CHANNELS=68
PARAMETERS=15573314
class Field(OriginalField):
 def __init__(self):
  super().__init__("native")
  old_local=self.local[0]
  old_film=self.film
  old_stem=self.stem
  self.stem=nn.Conv2d(70*16,384,1)
  self.film=nn.Linear(68,768)
  self.local[0]=nn.Conv2d(68,64,1)
  with torch.no_grad():
   self.stem.weight.zero_();self.stem.weight[:,:13*16].copy_(old_stem.weight)
   self.stem.bias.copy_(old_stem.bias)
   self.film.weight.zero_();self.film.weight[:,:58].copy_(old_film.weight)
   self.film.bias.copy_(old_film.bias)
   self.local[0].weight.zero_();self.local[0].weight[:,:58].copy_(old_local.weight)
   self.local[0].bias.copy_(old_local.bias)
 def pack(self,x):
  if x.shape[1:]!=(68,256,256):raise ValueError("Expected raw68 native fields")
  return F.pixel_unshuffle(torch.cat((global_inputs(x),x[:,11:68].clamp(-12,12)),1),4)
def create(seed=123041):
 torch.manual_seed(seed);return Field()
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def inputs(basic,physical,baseline,legacy_fspl,diffraction):
 if basic.shape[1]!=11 or physical.shape[1]!=45 or diffraction.shape[1]!=10:raise ValueError("Input contract changed")
 if baseline.shape!=legacy_fspl.shape or baseline.shape[1]!=2:raise ValueError("C82 pair required")
 return torch.cat((basic,physical,(baseline-legacy_fspl)/20,diffraction),1).contiguous(memory_format=torch.channels_last)
