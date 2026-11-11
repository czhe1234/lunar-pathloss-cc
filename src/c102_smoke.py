"""Profile geometry-weighted attention, including explicit pair-kernel cost."""
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c102_transport import Transport,geometry
g=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(102041)
results=[]
for scale in [4,1]:
 b=8;q=128;k=(256//scale)**2
 net=Transport().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,fused=True)
 h=torch.zeros(b,256,256,device='cuda');comp=torch.zeros(b,45,256,256,device='cuda');comp[:,16]=1
 tr=torch.full((b,),128,device='cuda',dtype=torch.long);tc=tr.clone()
 pixels=torch.randint(0,65536,(b,q),device='cuda')
 query=torch.randn(b,q,681,device='cuda');nodes=torch.randn(b,k,106,device='cuda');y=torch.randn(b,q,2,device='cuda')
 times=[];torch.cuda.reset_peak_memory_stats()
 for step in range(9):
  torch.cuda.synchronize();t=time.monotonic();bias,mass=geometry(h,tr,tc,pixels,comp,scale)
  assert torch.isfinite(bias).all() and torch.isfinite(mass).all()
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',torch.bfloat16):p=net(query,nodes,bias,mass)
  loss=(p-y).square().mean();loss.backward();assert torch.isfinite(loss);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
 row=dict(scale=scale,source_nodes=k,batch=b,queries_per_map=q,seconds_per_step=float(np.mean(times[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
 print(row,flush=True);results.append(row);del net,opt,h,comp,tr,tc,pixels,query,nodes,y,bias,mass,p,loss;torch.cuda.empty_cache()
a=R/'audit/c102_transport';a.mkdir(exist_ok=True);atomic_json(a/'smoke.json',dict(GPU=g,results=results));print('C102 SMOKE PASSED',flush=True)
