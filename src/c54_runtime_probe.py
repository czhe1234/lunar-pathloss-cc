"""Compare deployment precision/chunk sizes against frozen canonical outputs."""
import sys,csv,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c54_inference import Predictor
import numpy as np,torch
gpu=gpu_guard();net=Predictor(R/'runs/c54_combo_dev')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
records=[]
for i in [0,154]:
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);r=int(row['tx_row']);c=int(row['tx_col'])
 net.chunk=512;net.precision='fp32';ref=net.predict(h,r,c)
 def score(pr):
  result=[]
  for b in range(2):
   keep=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool);e=pr[b][keep].astype(float)-Y[b][i][keep].astype(float)
   result.append(float(np.sqrt(np.mean(e*e))))
  return np.array(result)
 ref_rm=score(ref)
 for chunk,precision in [(2048,'fp32'),(8192,'fp32'),(16384,'fp32'),(4096,'bf16'),(8192,'bf16'),(16384,'bf16')]:
  net.chunk=chunk;net.precision=precision;times=[]
  for k in range(3):
   torch.cuda.synchronize();t=time.perf_counter();pr=net.predict(h,r,c);torch.cuda.synchronize();times.append(time.perf_counter()-t)
  diff=pr.astype(float)-ref.astype(float);rm=score(pr);delta=rm-ref_rm
  rec={'map':i,'chunk':chunk,'precision':precision,'warm_mean_seconds_both_maps':float(np.mean(times[1:])),'all_seconds':times,
       'RMS_difference_db':np.sqrt((diff*diff).mean((1,2))).tolist(),'maximum_abs_difference_db':float(abs(diff).max()),
       'RMSE_change_vs_FP32_canonical_db':delta.tolist(),'RMSE':rm.tolist(),'canonical_RMSE':ref_rm.tolist()}
  if precision=='fp32':assert rec['maximum_abs_difference_db']<.005,rec
  else:assert max(abs(delta))<.01,rec
  records.append(rec);print(json.dumps(rec),flush=True)
atomic_json(R/'audit/c54_band_experts_20261102/runtime_probe.json',{'gpu':gpu,'records':records,
 'scope':'SharedGPU; rawH/TX public interface timed; FIT labels read only for numerical precision scoring, never predictor input',
 'goal':'Released README requires500ms/map. Both-map timing reported conservatively.'})
