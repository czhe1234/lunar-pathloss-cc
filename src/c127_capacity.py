import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c127_train import create,group_batch,evaluate,reader,Panel
from src.c127_joint import objective
A=R/'audit/c127_joint'
g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
plan=json.loads((A/'protocol.json').read_text());groups=[u for u in plan['groups'].values() if len(u)==4][:4];ids=[plan['fit'][j] for u in groups for j in u]
data=Panel(reader(),ids,reference=True);net=create('direct').cuda().to(memory_format=torch.channels_last)
out=R/'runs/c127_capacity_direct_joint';out.mkdir(exist_ok=False)
cfg=dict(GPU=g,ids=ids,scope='FIT-only16maps/4terrains,capacity diagnostic,not heldout result',steps=3000,lr=.0004,weight_decay=.01,mode='direct_joint',official_val_used=False)
atomic_json(out/'config.json',cfg);opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True)
ix=list(range(16));idx=torch.tensor(ix,device='cuda');group,first,inverse=group_batch(data,ix);x,ref,y,v=data.batch(idx);weight=torch.ones(16,device='cuda')
history=[];started=time.monotonic()
for step in range(1,3001):
 with torch.autocast('cuda',torch.bfloat16):s,t=net.components(x,first,inverse)
 loss,parts,ss,cc=objective(s,t,ref,y,v,weight,group,'joint')
 opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
 if step%100==0:
  metric=evaluate(net,data,'direct',16)
  rec=dict(step=step,FIT=metric,loss_parts={k:float(z) for k,z in parts.items()},seconds=time.monotonic()-started)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec)
  print('FIT',json.dumps(dict(step=step,rmse=metric['rmse'],tail=metric['tail_rmse'],seconds=rec['seconds'])),flush=True)
  if max(metric['rmse'])<.95:break
torch.save(dict(state=net.state_dict(),step=step,config=cfg),out/'last.pt')
metric=evaluate(net,data,'direct',16,save=out/'fit_prediction.npy')
atomic_json(out/'result.json',dict(status='complete',step=step,FIT=metric,fit_aggregate_gate=max(metric['rmse'])<1.,scope=cfg['scope'],seconds=time.monotonic()-started))
print('DONE',json.dumps(dict(step=step,FIT=metric['rmse'],tail=metric['tail_rmse'])),flush=True)
