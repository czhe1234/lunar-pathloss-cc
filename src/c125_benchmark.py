import sys,time,json,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c125_net import Panel,loss,A
from src.c125_neural import restore
import numpy as np,torch
g=gpu_guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d=Panel('fit');out=[]
for batch in [1024,2048,4096,8192]:
 net,_=restore();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True);torch.cuda.reset_peak_memory_stats();t=[]
 for j in range(10):
  ix=torch.randint(d.n,(batch,),device='cuda');torch.cuda.synchronize();st=time.monotonic()
  opt.zero_grad(set_to_none=True);v,_=loss(net,d,ix,'raw');v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
  torch.cuda.synchronize();t.append(time.monotonic()-st);assert torch.isfinite(v)
 out.append(dict(batch=batch,seconds_per_step=float(np.mean(t[3:])),points_per_second=batch/float(np.mean(t[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30));print(out[-1],flush=True)
 del net,opt;gc.collect();torch.cuda.empty_cache()
atomic_json(A/'benchmark.json',dict(GPU=g,throughput=out,weights_discarded=True))
