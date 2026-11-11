"""Native-resolution attention identity, gradients, throughput; no heldout labels."""
import sys,time,json,hashlib,ast
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c100_native_attention import initialize
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
g=gpu_guard();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
parent=torch.load(R/'runs/c99_global/best.pt',map_location='cpu',weights_only=False)['state']
torch.manual_seed(100049);x=torch.randn(2,56,256,256,device='cuda');x[:,5]=-.5;x[:,6]=-.5
initial=[];hashes=[]
for mode in ['continue','local','global']:
 net=initialize(mode,parent).cuda().to(memory_format=torch.channels_last);hashes.append(digest(net))
 with torch.inference_mode(),torch.autocast('cuda',torch.bfloat16):initial.append(net(x).float())
 del net;torch.cuda.empty_cache()
assert len(set(hashes))==1
identity=[float((a-initial[0]).abs().max()) for a in initial];assert max(identity)==0,identity
del x,initial
records=[]
prior=R/'logs/c100_smoke_initial.log'
if prior.exists():
 records=[ast.literal_eval(l) for l in prior.read_text().splitlines() if l.startswith("{'mode':")]
done={(r['mode'],r['batch']) for r in records}
for mode in ['continue','local','global']:
 for batch in [8,16]:
  if (mode,batch) in done:continue
  net=initialize(mode,parent).cuda().to(memory_format=torch.channels_last)
  opt=torch.optim.AdamW(net.parameters(),lr=.0001,weight_decay=.01,fused=True)
  x=torch.randn(batch,56,256,256,device='cuda').contiguous(memory_format=torch.channels_last);x[:,5]=-.5;x[:,6]=-.5
  y=torch.randn(batch,2,256,256,device='cuda');torch.cuda.reset_peak_memory_stats();times=[]
  for k in range(9):
   torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',torch.bfloat16):q=net(x)
   loss=(q.float()-y).square().mean();loss.backward()
   assert torch.isfinite(loss)
   if mode!='continue' and k==8:assert net.extra.row.qkv.weight.grad.abs().max()>0
   opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
  rec=dict(mode=mode,batch=batch,seconds_per_step=float(np.mean(times[3:])),maps_per_second=batch/float(np.mean(times[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  print(rec,flush=True);records.append(rec);del net,opt,x,y,q,loss;torch.cuda.empty_cache()
p=R/'audit/c100_native_attention';p.mkdir(parents=True,exist_ok=True)
atomic_json(p/'smoke.json',dict(GPU=g,initial_modes_exact_equal=True,identity_max_difference=identity,throughput=records,note='Architecture benchmark uses one loaded C99 checkpoint; final common parent selected only after C99 run completes. No heldout labels.'))
print('C100 SMOKE PASSED',flush=True)
