"""Fixed unseen-pixel evaluation within FIT terrains, before C38 training."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np,lightgbm as lgb
from src.c34_model import Model
from src.gpu_forest import Forest
gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
D=R/'features/c38_dense';maps=len(np.load(D/'fit_map_indices.npy'));idx=(np.arange(maps)[:,None]*1024+np.arange(256,288)[None]).reshape(-1)
X=torch.from_numpy(np.array(np.load(D/'fit_x.npy',mmap_mode='r')[idx])).cuda()
Y=torch.from_numpy(np.array(np.load(D/'fit_y.npy',mmap_mode='r')[idx])).cuda()
FS=torch.from_numpy(np.array(np.load(D/'fit_fspl.npy',mmap_mode='r')[idx])).cuda()
M=torch.from_numpy(np.array(np.load(D/'fit_mask.npy',mmap_mode='r')[idx],np.float32)).cuda()
names=['C10','c34_periodic8','c35_periodic8','c37_wide_periodic8'];results={}
@torch.inference_mode()
def evaluate(name):
 if name=='C10':
  models=[Forest(lgb.Booster(model_file=str(R/f'runs/c10_points_phase_geometry_s20261101/band{b}.txt'))) for b in ['415','58']]
 else:
  ck=torch.load(R/'runs'/name/'best.pt',map_location='cuda',weights_only=False);c=ck['config'];st=ck['model']
  model=Model(c['mode'],c['members'],c['width'],st['embedding.mean'],st['embedding.std'],st['embedding.edges'],st['bias'][0]*20).cuda();model.load_state_dict(st);model.eval()
 ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss)
 for lo in range(0,len(X),4096):
  x=X[lo:lo+4096]
  if name=='C10':q=torch.stack([m.predict(x) for m in models],1).float()
  else:
   with torch.autocast('cuda',dtype=torch.bfloat16):q=model(x).mean(1)*20
  e=(q+FS[lo:lo+4096]-Y[lo:lo+4096]).double();m=M[lo:lo+4096]
  ss+=(e.square()*m).sum(0);cc+=m.sum(0).double()
 return {'RMSE':(ss/cc).sqrt().cpu().tolist(),'valid_pixels':cc.cpu().tolist()}
t0=time.time()
for name in names:
 results[name]=evaluate(name);torch.cuda.empty_cache();print(name,results[name],flush=True)
out={'scope':'32 previously unseen query pixels in every FIT map; original256 excluded, no officialVAL','points':len(idx),'maps':maps,'selection':'first32 of768 new uniform C38 pixels; independent of labels and masks','C38_not_yet_trained':True,'models':results,'gpu':gpu,'seconds':time.time()-t0}
atomic_json(R/'audit/c38_density_continuation_20261101/unseen_FIT_pixels.json',out)
