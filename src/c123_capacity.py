"""Bounded 16-map fit and gradient diagnostic for the new architecture."""
import sys,json,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c105_context import reader,ResidualPanel
from src.c108_train import evaluate
from src.c123_lossless import create
import torch,numpy as np
A=R/'audit/c123_lossless'
ap=argparse.ArgumentParser();ap.add_argument('--eps',type=float,default=1e-8);a=ap.parse_args()
g=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
cfg=json.load(open(R/'audit/c105_context/protocol.json'));ids=cfg['fit'][:16];p=ResidualPanel(reader(),ids,'context')
net=create('native').cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,eps=a.eps,fused=True)
out=R/'runs'/f'c123_capacity_eps{a.eps:g}';out.mkdir(exist_ok=False);initial=evaluate(net,p,8,force_base=True);start=time.monotonic();history=[];grads=[]
atomic_json(out/'config.json',dict(ids=ids,parameters=sum(p.numel() for p in net.parameters()),GPU=g,eps=a.eps,max_steps=1200,criterion='both training RMSE<.95 only; no generalization claim',initial=initial))
gen=torch.Generator().manual_seed(123073)
for step in range(1,1201):
 ix=torch.randperm(16,generator=gen)[:8].cuda();x,b,y,w=p.batch(ix)
 with torch.autocast('cuda',torch.bfloat16):pred=b+net(x).float()
 err=torch.where(w,pred-y,0);loss=(err.square().sum((0,2,3))/w.sum((0,2,3))).mean()/25
 opt.zero_grad(set_to_none=True);loss.backward()
 if step in [1,2,10,100,400]:
  rec={'step':step,'layers':{}}
  for name,param in net.named_parameters():
   if name in ['stem.weight','blocks.0.row.qkv.weight','blocks.7.row.qkv.weight','unpack.weight','local.0.weight','mix.weight','head.weight']:
    v=param.grad.float();rec['layers'][name]=dict(rms=float(v.square().mean().sqrt()),max=float(v.abs().max()),fraction_below_eps=float((v.abs()<a.eps).float().mean()),zero_fraction=float((v==0).float().mean()))
  grads.append(rec);atomic_json(out/'gradients.json',grads)
 torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
 if step%100==0:
  val=evaluate(net,p,8);record=dict(step=step,fit=val['rmse'],seconds=time.monotonic()-start);history.append(record);atomic_json(out/'history.json',history);print('FIT',record,flush=True)
  if max(val['rmse'])<.95:break
torch.save(dict(state=net.state_dict(),step=step),out/'last.pt')
atomic_json(out/'result.json',dict(fit=val['rmse'],met_capacity=max(val['rmse'])<.95,step=step,seconds=time.monotonic()-start,GPU=g));print('DONE',val['rmse'],flush=True)
