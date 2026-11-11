"""Numerical and latency QA of frozen fullTRAIN refit; only FIT labels scored."""
import sys,time,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c54_inference import Predictor as Canonical
from src.c60_inference import Predictor as Fast
gpu=gpu_guard();slow=Canonical(R/'runs/c65_combo_full');fast=Fast(R/'runs/c65_combo_full')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
records=[]
for i in [0,154]:
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);r=int(row['tx_row']);c=int(row['tx_col']);ref=slow.predict(h,r,c);times=[];last=None;repeat=None
 for k in range(4):
  torch.cuda.synchronize();t=time.perf_counter();out=fast.predict(h,r,c);torch.cuda.synchronize();times.append(time.perf_counter()-t)
  if last is not None:repeat=float(np.max(abs(out-last)))
  last=out.copy()
 def score(pr):
  rr=[]
  for b in range(2):
   keep=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool);e=pr[b][keep].astype(float)-Y[b][i][keep].astype(float);rr.append(float(np.sqrt(np.mean(e*e))))
  return np.array(rr)
 diff=out.astype(float)-ref.astype(float);delta=score(out)-score(ref);assert max(abs(delta))<.01,delta;assert repeat==0,repeat
 rec={'map':i,'seconds_all':times,'warm_seconds_both_bands':float(np.mean(times[1:])),'RMS_diff_vs_canonical_db':np.sqrt((diff*diff).mean((1,2))).tolist(),'score_change_db':delta.tolist(),'repeat_max_abs_diff':repeat,'canonical_FIT_RMSE':score(ref).tolist(),'compiled_FIT_RMSE':score(out).tolist()}
 np.save(R/('audit/c65_full_refit_20261102/interface_'+('H100' if 'H100' in gpu['name'] else 'A800')+'_map'+str(i)+'.npy'),out)
 records.append(rec);print(json.dumps(rec),flush=True)
atomic_json(R/('audit/c65_full_refit_20261102/interface_'+('H100' if 'H100' in gpu['name'] else 'A800')+'.json'),{'complete':True,'gpu':gpu,'records':records,'rawH_TX_interface_verified':True,'numerical_quality_bound_db':.01,'warm_both_outputs_under500ms':all(r['warm_seconds_both_bands']<.5 for r in records),'limit':'Two FIT terrains on the recorded GPU; startup/compilation reported separately, official hardware unmeasured. No officialVAL labels read.'})
