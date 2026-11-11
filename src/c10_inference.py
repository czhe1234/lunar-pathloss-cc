"""Standalone C10 inference from one heightmap and one TX pixel.
No data index, target, mask or precomputed feature cache is opened.
Run via CC/run_gpu3.sh to restrict CUDA to the authorized physical GPU.
"""
import os,sys,hashlib
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
import numpy as np,torch,lightgbm as lgb
from scipy.ndimage import uniform_filter
SRC=Path(__file__).resolve().parent;sys.path.insert(0,str(SRC))
from cc_core import gpu_guard
from c3_geometry import geometry
from c4_geometry import extra_geometry
from gpu_forest import Forest
from c8_tx_context import terrain_fields,tx_features
from c7_surface_features import tx_visibility,terrain_fields as surface_fields
from c9_extra_gpu import profile_gpu,surface_gpu
from c10_phase_geometry import phase_features_gpu
class C10Predictor:
 def __init__(self,model_directory):
  self.device=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8)
  self.model_directory=Path(model_directory)
  self.models=[lgb.Booster(model_file=str(self.model_directory/f'band{b}.txt')) for b in ['415','58']]
  assert all(m.num_feature()==294 for m in self.models)
  self.forests=[Forest(m) for m in self.models]
  yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
  self.model_sha256={b:hashlib.sha256((self.model_directory/f'band{b}.txt').read_bytes()).hexdigest() for b in ['415','58']}
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
  pr=np.stack([m.predict(xxgpu).float().cpu().numpy()+fs[:,k] for k,m in enumerate(self.forests)],0).reshape(2,256,256)
  if not np.isfinite(pr).all():raise RuntimeError('Non-finite prediction')
  return pr

