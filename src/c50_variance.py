"""Measure the member-MSE variance term; CPU scoring, input-only predictions."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c34_model import Model
import numpy as np,torch
gpu=gpu_guard();torch.set_num_threads(8);torch.set_float32_matmul_precision('high');torch.cuda.set_per_process_memory_fraction(.2)
ck=torch.load(R/'runs/c38_1024_continue/best.pt',map_location='cuda',weights_only=False);s=ck['model'];c=ck['config']
net=Model(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval();net.load_state_dict(s)
out={}
for part,xd,yd in [('fit','c38_dense','c38_dense'),('dev','c10_points','c6_points')]:
 x=np.load(R/f'features/{xd}/{part}_x.npy',mmap_mode='r');y=np.load(R/f'features/{yd}/{part}_y.npy',mmap_mode='r');fs=np.load(R/f'features/{yd}/{part}_fspl.npy',mmap_mode='r');mask=np.load(R/f'features/{yd}/{part}_mask.npy',mmap_mode='r')
 ids=np.linspace(0,len(x)-1,65536,dtype=np.int64);qs=[]
 with torch.inference_mode():
  for lo in range(0,len(ids),4096):
   a=torch.from_numpy(np.array(x[ids[lo:lo+4096]])).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):q=net(a)
   qs.append(q.cpu().numpy()*20)
 q=np.concatenate(qs).astype(np.float64);target=np.asarray(y[ids],np.float64)-fs[ids].astype(np.float64);m=np.asarray(mask[ids],bool)
 mean=q.mean(1);err=(mean-target)**2;var=q.var(1)
 cnt=m.sum(0);mse=(err*m).sum(0)/cnt;variance=(var*m).sum(0)/cnt
 individual=(((q-target[:,None])**2).mean(1)*m).sum(0)/cnt
 assert np.max(abs(individual-mse-variance))<1e-9
 out[part]={'rows':len(ids),'RMSE_of_mean':np.sqrt(mse).tolist(),'mean_member_RMSE_proxy':np.sqrt(individual).tolist(),
  'variance_penalty_db2':variance.tolist(),'variance_fraction_of_mean_prediction_MSE':(variance/mse).tolist()}
A=R/'audit/c50_cooperative_20261102';A.mkdir(exist_ok=True);atomic_json(A/'variance.json',{'gpu':gpu,'decomposition':out})
print(json.dumps(out),flush=True)
