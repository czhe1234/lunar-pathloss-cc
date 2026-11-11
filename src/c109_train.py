"""Matched path-reader ordering test on the fixed128terrain dataset."""
import sys,json,time,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import digest,reader,ResidualPanel
from src.c109_ordered_ray import create
A=R/'audit/c109_architecture'
@torch.inference_mode()
def evaluate(net,panel,batch=16,force_base=False):
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=ss.clone();gs=torch.zeros(3,2,device='cuda',dtype=torch.float64);gc=gs.clone();bs=ss.clone()
 tp=ss.clone();fp=ss.clone();fn=ss.clone()
 for start in range(0,len(panel.ids),batch):
  x,b,y,w=panel.batch(slice(start,start+batch))
  if force_base:p=b
  else:
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
  e=torch.where(w,p-y,0.);r=y-b
  ss+=e.square().sum((0,2,3),dtype=torch.float64);cc+=w.sum((0,2,3));bs+=torch.where(w,r.square(),0.).sum((0,2,3),dtype=torch.float64)
  for k in range(3):
   g=w&((r.abs()<5) if k==0 else (r>=5) if k==1 else (r<=-5));gs[k]+=torch.where(g,e.square(),0.).sum((0,2,3),dtype=torch.float64);gc[k]+=g.sum((0,2,3))
  truth=r>=5;pred=(p-b)>=5
  tp+=(w&truth&pred).sum((0,2,3));fp+=(w&~truth&pred).sum((0,2,3));fn+=(w&truth&~pred).sum((0,2,3))
 net.train();return dict(rmse=(ss/cc).sqrt().tolist(),C82_rmse=(bs/cc).sqrt().tolist(),normalized_C82_SSE=(ss/bs.clamp_min(1e-20)).tolist(),
  sse=ss.tolist(),count=cc.tolist(),group_rmse=(gs/gc.clamp_min(1)).sqrt().tolist(),group_count=gc.tolist(),
  tp=tp.tolist(),fp=fp.tolist(),fn=fn.tolist(),precision=(tp/(tp+fp).clamp_min(1)).tolist(),recall=(tp/(tp+fn).clamp_min(1)).tolist())
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['ordered','permuted'],required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();a.n=128
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());subset=plan['sets'][str(a.n)];net=create(a.mode);h=digest(net);assert h==plan['initial_state_sha256']
 net=net.cuda();net.trunk.to(memory_format=torch.channels_last);net.path.to(memory_format=torch.channels_last_3d);net.readout.to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True)
 if a.smoke:
  r=reader();data=ResidualPanel(r,subset['fit'][:8],'context');timing=[];g=torch.Generator().manual_seed(108073)
  before=evaluate(net,data,force_base=True)
  xx,bb,_,_=data.batch(slice(0,2));net.eval()
  with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
   assert torch.count_nonzero(net(xx))==0
  prof=net.profile(xx);net.mode='permuted';shuf=net.profile(xx);net.mode=a.mode
  assert torch.equal(prof.index_select(2,net.permutation),shuf)
  del prof,shuf
  net.train()
  for step in range(12):
   ix=torch.randint(len(data.ids),(8,),generator=g).cuda();x,b,y,w=data.batch(ix);torch.cuda.synchronize();t0=time.monotonic()
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
   e=torch.where(w,p-y,0.);loss=(e.square().sum((0,2,3))/w.sum((0,2,3)).clamp_min(1)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();torch.cuda.synchronize()
   assert torch.isfinite(loss);timing.append(time.monotonic()-t0)
  after=evaluate(net,data);rec=dict(GPU=gpu,batch=8,before=before,after=after,seconds_per_update=float(np.mean(timing[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  atomic_json(A/'smoke.json',rec);print(json.dumps(rec),flush=True);return
 out=R/'runs'/('c109_'+a.mode);out.mkdir(exist_ok=False);t=time.monotonic();r=reader()
 train=ResidualPanel(r,subset['fit'],'context')
 panels={'fit':train,'same_geometry_newTX_common8':ResidualPanel(r,plan['same_terrain_newTX_common8'],'context'),
  'new_geometry32':ResidualPanel(r,plan['unseen_residual_fit_terrain32'],'context'),'monitor':ResidualPanel(r,plan['original_monitor'],'context')}
 config=dict(mode=a.mode,n=a.n,GPU=gpu,parameters=plan['parameters'],initial_state_sha256=h,batch=8,steps=plan['steps'],fit_maps=len(train.ids),fit_terrains=a.n,
  sample_seed=108073,initial_seed=105041,optimizer=plan['optimizer'],official_val_used=False)
 atomic_json(out/'config.json',config)
 initial={k:evaluate(net,p,force_base=True) for k,p in panels.items()};atomic_json(out/'initial.json',initial)
 best=float(np.mean(initial['monitor']['rmse']));beststep=0
 torch.save(dict(state=net.state_dict(),step=0,config=config),out/'best.pt')
 gen=torch.Generator().manual_seed(108073);history=[];print('INITIAL',a.n,{k:v['rmse'] for k,v in initial.items()},'seconds',time.monotonic()-t,flush=True)
 for step in range(1,plan['steps']+1):
  ix=torch.randint(len(train.ids),(8,),generator=gen).cuda();x,b,y,w=train.batch(ix)
  with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
  e=torch.where(w,p-y,0.);loss=(e.square().sum((0,2,3))/w.sum((0,2,3)).clamp_min(1)).mean()/25
  opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
  assert torch.isfinite(loss) and torch.isfinite(grad)
  if step in plan['eval_steps']:
   val={k:evaluate(net,p) for k,p in panels.items()};criterion=float(np.mean(val['monitor']['rmse']))
   ck=dict(state=net.state_dict(),step=step,config=config);torch.save(ck,out/f'step{step}.pt')
   if criterion<best:
    best=criterion;beststep=step;torch.save(ck,out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   row=dict(step=step,evaluation=val,best_step=beststep,loss=float(loss),gradient_norm=float(grad),seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
   history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',row)
   print('EVAL',a.n,json.dumps(row),flush=True)
   torch.save(dict(**ck,optimizer=opt.state_dict(),order=gen.get_state()),out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 selected={k:evaluate(net,p,force_base=beststep==0) for k,p in panels.items()}
 result=dict(status='complete',n=a.n,best_step=beststep,last_step=plan['steps'],selected=selected,last=history[-1]['evaluation'],
  seconds=time.monotonic()-t,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',a.n,beststep,{k:v['rmse'] for k,v in selected.items()},flush=True)
if __name__=='__main__':main()
