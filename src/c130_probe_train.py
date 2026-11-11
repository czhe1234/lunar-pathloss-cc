"""Matched conditional diagnostics; not a deployable all-pixel predictor."""
import sys,time,json,argparse,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c130_probe_model import Panel,create,loss,evaluate,A
from src.c105_context import digest
import torch,numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['old','nodes','shuffled']);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');start=time.monotonic()
 train=Panel('fit');net=create(train)
 if a.smoke:
  initial=digest(net);torch.save(dict(state=net.state_dict(),hash=initial),A/'probe_initial.pt')
  opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True);ix=torch.arange(2048,device='cuda');history=[];times=[]
  for step in range(12):
   torch.cuda.synchronize();t=time.monotonic();v=loss(net,train,ix,'nodes');opt.zero_grad(set_to_none=True);v.backward()
   grad=float(net.patch_in.weight.grad.norm());torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(v)
   torch.cuda.synchronize();times.append(time.monotonic()-t);history.append(float(v))
  assert grad>0 and history[-1]<history[0]
  atomic_json(A/'probe_smoke.json',dict(status='passed',GPU=gpu,initial_sha256=initial,parameters=sum(p.numel() for p in net.parameters()),loss_history=history,
   patch_gradient_norm=grad,seconds_per_step=float(np.mean(times[5:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30));print('SMOKE PASSED',history,flush=True);return
 assert json.loads((A/'probe_smoke.json').read_text())['status']=='passed'
 ck=torch.load(A/'probe_initial.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);assert digest(net)==ck['hash']
 mon=Panel('monitor');out=R/('runs/c130_nodes_'+a.mode);out.mkdir(exist_ok=False)
 config=dict(mode=a.mode,GPU=gpu,initial_sha256=ck['hash'],fit_points=train.n,monitor_points=mon.n,batch=2048,steps=4000,eval_interval=250,min_steps=1500,patience=1000,
  seed=130173,lr=.0003,weight_decay=.01,selection='minimummeanconditionalTRUETAILmonitorRMSE;labelsnotpredictorinput',objective='tailMSE/100+0.1signBCE; exactexpectationdenominator',
  shuffle_hashes=[train.perm_hash,mon.perm_hash],official_val_used=False,scope='diagnosticwithtruthselectedtailpoints;notdeployablescore',
  source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'src/c130_probe_model.py',R/'src/c130_nodes.py']})
 atomic_json(out/'config.json',config);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True);rng=np.random.default_rng(130173)
 initial=evaluate(net,mon,a.mode);best=float(np.mean(initial['tail_RMSE']));best_step=0;hist=[]
 atomic_json(out/'initial.json',initial);torch.save(dict(state=net.state_dict(),step=0,config=config),out/'best.pt')
 for step in range(1,4001):
  ix=torch.tensor(rng.integers(train.n,size=2048),device='cuda',dtype=torch.long)
  lr=.0003*min(1.,.1+.9*step/100)*(.3+.7*.5*(1+math.cos(math.pi*step/4000)))
  for pg in opt.param_groups:pg['lr']=lr
  v=loss(net,train,ix,a.mode);opt.zero_grad(set_to_none=True);v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(v)
  if step%250==0:
   val=evaluate(net,mon,a.mode);fit=evaluate(net,train,a.mode);criterion=float(np.mean(val['tail_RMSE']))
   if criterion<best:
    best=criterion;best_step=step;torch.save(dict(state=net.state_dict(),step=step,config=config),out/'best.pt')
   row=dict(step=step,FIT=fit,monitor=val,best_step=best_step,seconds=time.monotonic()-start)
   hist.append(row);atomic_json(out/'history.json',hist);atomic_json(out/'progress.json',row);print('STEP',json.dumps(row),flush=True)
   torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),step=step,config=config),out/'last.pt')
   if step>=1500 and step-best_step>=1000:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 result=dict(status='complete',mode=a.mode,best_step=best_step,last_step=step,FIT=evaluate(net,train,a.mode,save=out/'fit'),
  monitor=evaluate(net,mon,a.mode,save=out/'monitor'),seconds=time.monotonic()-start,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
