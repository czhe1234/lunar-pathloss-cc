"""Cache frozen internal-FIT C38 predictions. No targets or validity masks opened."""
import sys,time,json,hashlib,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c34_model import Model
p=argparse.ArgumentParser();p.add_argument('--part',choices=['fit','dev'],required=True);a=p.parse_args()
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.5);torch.set_float32_matmul_precision('high')
o=R/'features/c47_teacher';o.mkdir(exist_ok=True)
ckp=R/'runs/c38_1024_continue/best.pt';ck=torch.load(ckp,map_location='cuda',weights_only=False);s=ck['model'];c=ck['config']
net=Model(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval();net.load_state_dict(s)
xp=R/('features/c38_dense/fit_x.npy' if a.part=='fit' else 'features/c10_points/dev_x.npy')
x=np.load(xp,mmap_mode='r');dest=o/(a.part+'_base.npy');assert not dest.exists()
out=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(x),2));t=time.time()
with torch.inference_mode():
 for lo in range(0,len(x),4096):
  v=torch.from_numpy(np.array(x[lo:lo+4096],copy=True)).cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(v).mean(1)
  out[lo:lo+len(v)]=q.float().cpu().numpy()
  if lo%(4096*100)==0:
   z={'part':a.part,'rows':min(lo+len(v),len(x)),'total':len(x),'seconds':time.time()-t};atomic_json(o/(a.part+'_progress.json'),z);print(json.dumps(z),flush=True)
out.flush();assert np.isfinite(out).all()
meta={'complete':True,'part':a.part,'rows':len(x),'shape':list(out.shape),'target':'(prediction PL-FSPL)/20',
 'input_only':True,'labels_masks_read':False,'X_source':str(xp),'teacher':str(ckp),
 'teacher_sha256':hashlib.sha256(ckp.read_bytes()).hexdigest(),'gpu':gpu,'seconds':time.time()-t}
atomic_json(o/(a.part+'_meta.json'),meta);print(json.dumps(meta),flush=True)
