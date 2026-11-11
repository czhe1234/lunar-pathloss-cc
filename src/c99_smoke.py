"""Check context reach, same weights, actual backward and measured batch throughput."""
import sys,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c99_axial import AxialField,Axis
def digest(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 g=gpu_guard();torch.set_num_threads(6);torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
 a=Axis(8,64,False,False);b=Axis(8,64,False,True)
 with torch.no_grad():
  for t in [a,b]:
   t.qkv.weight.zero_();t.qkv.bias.zero_();t.qkv.weight[16:].copy_(torch.eye(8));t.out.weight.copy_(torch.eye(8));t.out.bias.zero_();t.scale.fill_(1)
  x=torch.zeros(1,8,1,64);x[0,0,0,40]=1;x[0,1,0,40]=-1
  local=float(a(x)[0,:,0,0].abs().max());global_=float(b(x)[0,:,0,0].abs().max())
 assert local==0 and global_>.001,(local,global_)
 torch.manual_seed(99041);a=AxialField(False);ha=digest(a)
 torch.manual_seed(99041);b=AxialField(True);hb=digest(b);assert ha==hb
 params=sum(p.numel() for p in b.parameters());del a,b
 results=[]
 for mode in ['local','global']:
  for batch in [4,8,16,32]:
   torch.manual_seed(99041);net=AxialField(mode=='global').cuda().to(memory_format=torch.channels_last)
   opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.01,fused=True)
   x=torch.randn(batch,56,256,256,device='cuda').contiguous(memory_format=torch.channels_last)
   x[:,5]=-.5;x[:,6]=-.5
   y=torch.randn(batch,2,256,256,device='cuda')
   torch.cuda.reset_peak_memory_stats();times=[]
   for k in range(9):
    torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',torch.bfloat16):q=net(x)
    loss=(q.float()-y).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if k==8:assert net.stage1[0].row.qkv.weight.grad.abs().max()>0
    opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
   rec=dict(mode=mode,batch=batch,seconds_per_step=float(np.mean(times[3:])),maps_per_second=batch/float(np.mean(times[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
   results.append(rec);print(rec,flush=True)
   del net,opt,x,y,q,loss;torch.cuda.empty_cache()
 out=dict(gpu=g,context_check=dict(local_remote_response=local,global_remote_response=global_),same_initial_state_sha256=ha,parameters=params,throughput=results)
 (R/'audit/c99_axial').mkdir(parents=True,exist_ok=True)
 atomic_json(R/'audit/c99_axial/smoke.json',out)
 print('SMOKE PASSED',params,ha,flush=True)
if __name__=='__main__':main()
