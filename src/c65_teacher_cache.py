"""Input-only full-TRAIN frozen C38 teacher cache for selected C54 refit."""
import sys,time,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c34_model import Model
from src.c47_train import device_array
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.7);torch.set_float32_matmul_precision('high')
path=R/'runs/c38_full_refit/best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['model']
net=Model('periodic',s['head'].shape[0],s['head'].shape[1],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval();net.load_state_dict(s)
folder=R/'features/c65_full_teacher';folder.mkdir(exist_ok=False);t=time.time()
for part,source in [('fit','c38_dense'),('dev','c38_dev_full')]:
 x=device_array(R/f'features/{source}/{part}_x.npy',dtype=torch.float32)
 out=np.lib.format.open_memmap(folder/(part+'_base.npy'),mode='w+',dtype=np.float32,shape=(len(x),2))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for lo in range(0,len(x),4096):out[lo:lo+4096]=net(x[lo:lo+4096]).mean(1).float().cpu().numpy()
 out.flush();print(json.dumps({'part':part,'rows':len(x),'seconds':time.time()-t}),flush=True);del x,out;torch.cuda.empty_cache()
atomic_json(folder/'meta.json',{'complete':True,'input_only':True,'labels_masks_read':False,'teacher':str(path.relative_to(R)),'teacher_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'batch':4096,'precision':'BF16','scope':'All6120TRAINterrains. Original internalDEV part is included in fullTRAIN refit; officialVAL excluded.','gpu':gpu,'seconds':time.time()-t})
