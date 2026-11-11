"""C54 compile/fusion speed check; two different FIT terrains catch stale inputs."""
import sys,time,csv,json,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c54_inference import Predictor
from src.c3_geometry import geometry
class FastPredictor(Predictor):
 @torch.inference_mode()
 def set_terrain(self,H,tr,tc):
  h=torch.from_numpy(np.array(H,np.float32,copy=True)).cuda();sy,sx=torch.gradient(h,dim=(-2,-1))
  self.high.fields.copy_(torch.stack((h,sx,sy),-1)[None])
  self.high.map_info.copy_(torch.tensor([[0,int(tr),int(tc)]],device='cuda',dtype=torch.long))
  g=geometry(h[None],torch.tensor([tr],device='cuda'),torch.tensor([tc],device='cuda'))[0,:8].float()
  t=(torch.arange(8,device='cuda',dtype=torch.float32)+.5)/8;b=2*torch.sinh(3*g)
  self.high.margins.copy_(torch.stack(((b+.95*t[:,None,None]).amax(0),(b+.95*t[:,None,None]+(1-t[:,None,None])).amax(0)),-1).half()[None])

class Predictor(FastPredictor):
 def __init__(self,model_directory):
  super().__init__(model_directory);self.chunk=4096;self.precision='bf16'
  for mod in [self.teacher,self.low,self.high]:mod.forward=torch.compile(mod.forward,mode='default',fullgraph=True,dynamic=False)
