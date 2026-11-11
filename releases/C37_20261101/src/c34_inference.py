"""Standalone numerical-embedding network inference from one heightmap and one TX pixel.
No data index, target, mask or precomputed feature cache is opened.
Run via CC/run_gpu.sh {1|3|4}; the existing UUID guard verifies the device.
"""
import os,sys,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.ndimage import uniform_filter
SRC=Path(__file__).resolve().parent;sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC.parent))
from cc_core import gpu_guard
from c3_geometry import geometry
from c4_geometry import extra_geometry
from src.c34_model import Model
from c8_tx_context import terrain_fields,tx_features
from c7_surface_features import tx_visibility,terrain_fields as surface_fields
from c9_extra_gpu import profile_gpu,surface_gpu
from c10_phase_geometry import phase_features_gpu
class Predictor:
 def __init__(self,model_directory):
  self.device=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8)
  self.model_directory=Path(model_directory)
  ck=torch.load(self.model_directory/'best.pt',map_location='cuda',weights_only=False)
  cfg=ck['config'];st=ck['model'];cls=Model
  if cfg.get('expanded',False):
   from src.c36_model import Model as cls
  self.net=cls(cfg['mode'],cfg['members'],cfg['width'],st['embedding.mean'],st['embedding.std'],st['embedding.edges'],st['bias'][0]*20).cuda()
  self.net.load_state_dict(st);self.net.eval();torch.set_float32_matmul_precision('high')
  yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
 @torch.inference_mode()
 def predict(self,heightmap,tx_row,tx_col):
  h=np.array(heightmap,dtype=np.float32,copy=True);tr=int(tx_row);tc=int(tx_col)
  if h.shape!=(256,256) or not np.isfinite(h).all():raise ValueError('Expected a finite 256x256 heightmap in metres')
  if tr!=tx_row or tc!=tx_col or not (0<=tr<256 and 0<=tc<256):raise ValueError('TX must be an integer pixel inside the map')
  tf=terrain_fields(h);sx=tf[1];sy=tf[2];hp=h-uniform_filter(h,size=5,mode='nearest')
  yy=self.yy;xx=self.xx;ht=float(h[tr,tc]);dx=xx-tc;dy=yy-tr;dz=h+1-ht-3;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
  basic=np.stack([h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((yy==tr)&(xx==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)],0).reshape(11,-1).T
  hh=torch.from_numpy(h[None]).cuda();rr=torch.tensor([tr],device='cuda');cc=torch.tensor([tc],device='cuda')
  fg=torch.cat((geometry(hh,rr,cc),extra_geometry(hh,rr,cc)),1)[0].float().cpu().numpy();gg=fg.reshape(20,-1).T
  ctx=tx_features(tf,tr,tc);x=np.concatenate((basic,gg,np.broadcast_to(ctx,(65536,len(ctx)))),1)
  fs=(20*np.log10(4*np.pi*dist.reshape(-1,1)*np.array([415e6,5.775e9])[None]/299792458.)).astype(np.float32)
  crit=2.5+2*np.sinh(3*x[:,27]);phase=np.arange(8)/8;txh=x[:,31]
  extra=np.arcsinh((crit[:,None]-(3-np.remainder(txh[:,None]+phase,1)))/2)/3
  hp_gpu=hh[0];pv=profile_gpu(hp_gpu,tr,tc);sv=surface_gpu(hp_gpu,surface_fields(h),[tx_visibility(fg[:16],ht_) for ht_ in [3,2]],tr,tc)
  x=np.concatenate((x,pv.cpu().numpy(),sv.cpu().numpy(),extra.astype(np.float32)),1)
  xxgpu=torch.from_numpy(x).cuda();xxgpu=torch.cat((xxgpu,phase_features_gpu(xxgpu)),1)
  out=[]
  for lo in range(0,len(xxgpu),4096):
   with torch.autocast('cuda',dtype=torch.bfloat16):q=self.net(xxgpu[lo:lo+4096]).mean(1)
   out.append(q.float().cpu().numpy()*20)
  pr=(np.concatenate(out)+fs).T.reshape(2,256,256)
  if not np.isfinite(pr).all():raise RuntimeError('Non-finite prediction')
  return pr

