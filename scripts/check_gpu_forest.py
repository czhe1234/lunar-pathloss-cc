import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard,atomic_json
info=gpu_guard()
import torch,numpy as np,lightgbm as lgb
from gpu_forest import Forest
torch.cuda.set_per_process_memory_fraction(.15)
torch.set_num_threads(8)
run=R/'runs/c8_points_context_s20261101';x=np.array(np.load(R/'features/c8_points/dev_x.npy',mmap_mode='r')[:65536],copy=True)
xx=torch.from_numpy(x).cuda();results=[]
for b in ['415','58']:
 m=lgb.Booster(model_file=str(run/f'band{b}.txt'));f=Forest(m)
 t=time.time();a=f.predict(xx);torch.cuda.synchronize();cold=time.time()-t
 cpu=m.predict(x,num_threads=16);gpu=a.cpu().numpy();err=float(abs(cpu-gpu).max());assert err<1e-9,err
 times=[]
 for _ in range(5):
  torch.cuda.synchronize();t=time.perf_counter();y=f.predict(xx);torch.cuda.synchronize();times.append(time.perf_counter()-t)
 rec={'band':b,'max_abs_CPU_GPU_float64':err,'float32_exports_equal':bool(np.array_equal(cpu.astype(np.float32),gpu.astype(np.float32))),'cold_seconds':cold,'warm_seconds':times,'trees':f.ntrees,'nodes':f.nodes,'depth':f.depth}
 results.append(rec);print(json.dumps(rec),flush=True)
out={'GPU':info,'memory_fraction_cap':.15,'peak_GB':torch.cuda.max_memory_allocated()/1e9,'points':len(x),'results':results}
atomic_json(R/'audit/gpu_forest_verification.json',out);print(json.dumps(out),flush=True)

