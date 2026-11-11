import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from torch.nn import functional as F
from src.c105_context import reader,ResidualPanel,digest
from src.c123_lossless import create,global_inputs
A=R/'audit/c123_lossless'
g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
plan=json.load(open(R/'audit/c105_context/protocol.json'));r=reader();panel=ResidualPanel(r,plan['fit'][:16],'context');x,b,y,w=panel.batch(slice(0,2))
records=[]
for mode in ['native','mean']:
 n=create(mode).cuda().to(memory_format=torch.channels_last);h=digest(n);raw=global_inputs(x);back=F.pixel_shuffle(n.pack(x),4)
 expected=raw if mode=='native' else F.avg_pool2d(raw,4).repeat_interleave(4,2).repeat_interleave(4,3)
 assert torch.equal(back,expected)
 with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):out=n(x)
 assert torch.equal(out,torch.zeros_like(out))
 records.append(dict(mode=mode,parameters=sum(p.numel() for p in n.parameters()),initial_sha=h,pack_unpack_max_abs=float((back-expected).abs().max()),initial_prediction_exact_C82=True))
 del n,raw,back,out
assert records[0]['initial_sha']==records[1]['initial_sha']
n=create('native').cuda().to(memory_format=torch.channels_last)
# A remote source pixel reaches a receiver outside the local branch's receptive field.
with torch.no_grad():n.head.weight.normal_(std=.01)
xx=x[:1].detach().clone().requires_grad_(True);z=n(xx);loss=z[0,:,220,220].square().sum();loss.backward()
remote=float(xx.grad[0,0,20,20].abs());assert remote>0
del n,xx,z,loss;torch.cuda.empty_cache()
speeds=[]
for batch in [4,8,16]:
 net=create('native').cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True)
 xx,bb,yy,ww=panel.batch(slice(0,batch));tt=[];torch.cuda.reset_peak_memory_stats()
 for j in range(7):
  torch.cuda.synchronize();t=time.monotonic()
  with torch.autocast('cuda',torch.bfloat16):p=bb+net(xx).float()
  e=torch.where(ww,p-yy,0);loss=(e.square().sum((0,2,3))/ww.sum((0,2,3))).mean()/25
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();torch.cuda.synchronize()
  assert torch.isfinite(loss);tt.append(time.monotonic()-t)
 backbone_grad=float(net.stem.weight.grad.abs().max());assert backbone_grad>0
 speeds.append(dict(batch=batch,seconds=float(np.mean(tt[2:])),maps_per_second=batch/float(np.mean(tt[2:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30,backbone_gradient_max=backbone_grad))
 print('BENCH',speeds[-1],flush=True);del net,opt,xx,bb,yy,ww,p,e,loss;torch.cuda.empty_cache()
result=dict(GPU=g,models=records,remote_gradient=remote,speed=speeds,labels_and_loss='FP32',math='BF16 main matrix operations')
atomic_json(A/'smoke.json',result);print('SMOKE_DONE',json.dumps(result),flush=True)
