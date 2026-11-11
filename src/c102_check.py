"""Real-data alignment, zero-correction and gradient checks for the transport operator."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch
from src.c102_train import load_encoders
from src.c102_data import Bank
from src.c102_transport import Transport
gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
point,source=load_encoders();bank=Bank('fit',source,limit=8)
maps=torch.arange(8,device='cuda');pos=torch.arange(128,device='cuda')[None].expand(8,-1)
records=[]
for scale in [4,1]:
 query,nodes,bias,mass,base,t,w=bank.batch(maps,pos,scale,point)
 assert query.shape==(8,128,681) and nodes.shape[2]==106 and torch.isfinite(bias).all()
 torch.manual_seed(102041);net=Transport().cuda()
 with torch.autocast('cuda',torch.bfloat16):delta=net(query,nodes,bias,mass)
 assert torch.count_nonzero(delta)==0
 opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.005,fused=True)
 for step in range(3):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',torch.bfloat16):p=base+net(query,nodes,bias,mass)
  mask=t[...,2:4];loss=(((p-t[...,:2]).square()*mask).sum((0,1))/mask.sum((0,1)).clamp_min(1)).mean()/25
  loss.backward();assert torch.isfinite(loss);opt.step()
 assert net.q.weight.grad.abs().max()>0 and net.kv.weight.grad.abs().max()>0
 records.append(dict(scale=scale,source_nodes=nodes.shape[1],loss=float(loss),connected_gradients=True))
 del net,opt;torch.cuda.empty_cache()
a=R/'audit/c102_transport';atomic_json(a/'real_data_check.json',dict(passed=True,GPU=gpu,checks=records,labels_masks_in_input=False))
print('C102 DATA CHECK PASSED',records,flush=True)
