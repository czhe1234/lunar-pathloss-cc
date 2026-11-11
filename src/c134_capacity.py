"""Bounded 16-map capacity diagnostic on retained FIT query points only."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c134_data import Dataset
from src.c133_model import ContextTransport
from src.c134_train import score
import torch,numpy as np
A=R/'audit/c134_paired'
def main():
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.manual_seed(134041)
 plan=json.loads((A/'protocol.json').read_text());data=Dataset('fit',plan['fit'][:16]);net=ContextTransport('direction',3).cuda();data.neutral_cache(net)
 opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0,fused=True);history=[];initial=score(net,data);t=time.monotonic();ix=list(range(16))
 for step in range(1,2001):
  px=torch.randint(data.pp,(16,384),device='cuda');g,q,b,y,v=data.batch(ix,px);p=net(g,q,b)
  loss=(torch.where(v,(p-y).square(),0).sum((0,1))/v.sum((0,1))).mean()/25
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();assert torch.isfinite(loss)
  if step%100==0:
   val=score(net,data);rec=dict(step=step,fit=val,seconds=time.monotonic()-t);history.append(rec);atomic_json(A/'capacity_progress.json',rec);print('CAPACITY',json.dumps(rec),flush=True)
   if max(val['RMSE'])<1:break
 torch.save(dict(state=net.state_dict(),ids=data.ids,steps=step),A/'capacity16.pt')
 val=score(net,data,save=A/'capacity16_prediction.npy')
 atomic_json(A/'capacity16.json',dict(status='complete',GPU=gpu,initial=initial,final=val,history=history,ids=data.ids,steps=step,seconds=time.monotonic()-t,scope='16 FIT maps x1536 retained paired queries; not full maps, not holdout or officialVAL',monitor_used=False))
if __name__=='__main__':main()
