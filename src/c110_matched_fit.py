"""Complementary diversity control: match fitting on the identical24-map anchor."""
import sys,json,time,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import create,digest,reader,ResidualPanel
from src.c108_train import evaluate
A=R/'audit/c110_matched_fit'
def main():
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());data=json.loads((R/plan['data_protocol']).read_text());r=reader()
 anchor=ResidualPanel(r,data['sets']['8']['fit'],'context')
 test={'same_geometry_newTX_common8':ResidualPanel(r,data['same_terrain_newTX_common8'],'context'),
  'new_geometry32':ResidualPanel(r,data['unseen_residual_fit_terrain32'],'context'),
  'monitor':ResidualPanel(r,data['original_monitor'],'context')}
 for n in [8,32,128]:
  net=create();assert digest(net)==data['initial_state_sha256'];net=net.cuda().to(memory_format=torch.channels_last)
  opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True)
  train=ResidualPanel(r,data['sets'][str(n)]['fit'],'context');out=R/f'runs/c110_n{n}';out.mkdir(exist_ok=False);t=time.monotonic()
  config=dict(n=n,GPU=gpu,initial_state_sha256=data['initial_state_sha256'],sample_seed=108073,batch=8,max_steps=2000,
   stop='firstcommon24mapmeanrelativeC82SSE<=.85;checkpointusesFITonly',official_val_used=False)
  atomic_json(out/'config.json',config);baseline=evaluate(net,anchor,force_base=True);gen=torch.Generator().manual_seed(108073)
  history=[];passed=False
  interval={8:10,32:20,128:25}[n]
  for step in range(1,2001):
   ix=torch.randint(len(train.ids),(8,),generator=gen).cuda();x,b,y,w=train.batch(ix)
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
   e=torch.where(w,p-y,0.);loss=(e.square().sum((0,2,3))/w.sum((0,2,3)).clamp_min(1)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   if step%interval==0 or step==2000:
    fit=evaluate(net,anchor);ratio=float(np.mean(fit['normalized_C82_SSE']))
    rec=dict(step=step,anchor=fit,mean_ratio=ratio,seconds=time.monotonic()-t);history.append(rec);atomic_json(out/'progress.json',rec)
    if step%100==0 or ratio<=.85:print('ANCHOR',n,step,ratio,fit['rmse'],flush=True)
    if ratio<=.85:passed=True;break
  torch.save(dict(state=net.state_dict(),step=step,config=config),out/'selected.pt');atomic_json(out/'history.json',history)
  evaluation={k:evaluate(net,p) for k,p in test.items()};evaluation['fit_all']=evaluate(net,train)
  result=dict(status='complete',n=n,passed=passed,selected_step=step,anchor=history[-1]['anchor'],mean_anchor_ratio=history[-1]['mean_ratio'],
   evaluation=evaluation,seconds=time.monotonic()-t,GPU=gpu,official_val_used=False)
  atomic_json(out/'result.json',result);print('DONE',n,json.dumps(result),flush=True)
  del net,opt,train;torch.cuda.empty_cache()
if __name__=='__main__':main()
