import sys,json,time,argparse,math,hashlib,os,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c125_net import Panel,Net,create,digest,loss,evaluate,A
import numpy as np,torch
def restore():
 z=torch.load(A/'initial.pt',map_location='cpu',weights_only=False)
 net=Net(torch.zeros(169),torch.ones(169),torch.ones(2)*.5,torch.zeros(2)).cuda();net.load_state_dict(z['state']);return net,z
def smoke(g):
 train=Panel('fit');net=create(train);initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};ih=digest(net)
 torch.save(dict(state=initial,initial_sha256=ih),A/'initial.pt')
 records=[]
 for batch in [128,256,512]:
  net.load_state_dict(initial);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True)
  torch.cuda.reset_peak_memory_stats();times=[]
  for j in range(10):
   ix=torch.randint(train.n,(batch,),device='cuda');torch.cuda.synchronize();t=time.monotonic()
   opt.zero_grad(set_to_none=True);v,_=loss(net,train,ix,'raw');v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
   torch.cuda.synchronize();times.append(time.monotonic()-t);assert torch.isfinite(v)
  records.append(dict(batch=batch,seconds_per_step=float(np.mean(times[3:])),points_per_second=batch/float(np.mean(times[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30))
  del opt;gc.collect()
 # Cheap data/gradient check after nonzero head update.
 net.load_state_dict(initial);opt=torch.optim.AdamW(net.parameters(),lr=.0003,fused=True)
 for j in range(4):
  ix=torch.arange(256,device='cuda');opt.zero_grad(set_to_none=True);v,_=loss(net,train,ix,'raw');v.backward();opt.step()
 grads={k:float(p.grad.float().square().mean().sqrt()) for k,p in net.named_parameters() if k in ['number_in.weight','rx_in.weight','tx_in.weight','path_in.weight','head.weight']}
 assert all(np.isfinite(list(grads.values()))) and all(v>0 for v in grads.values())
 out=dict(GPU=g,initial_sha256=ih,parameters=sum(p.numel() for p in net.parameters()),throughput=records,gradient_RMS=grads,fit_points=train.n,shuffle_hash=train.perm_hash)
 atomic_json(A/'smoke.json',out);print(json.dumps(out),flush=True)
def run(a,g):
 plan=json.load(open(A/'protocol.json'))['neural'];out=R/f'runs/c125_nn_{a.mode}_fixed';out.mkdir(exist_ok=False)
 train=Panel('fit');monitor=Panel('monitor');net,initial=restore();assert digest(net)==initial['initial_sha256']==plan['initial_sha256']
 batch=plan['batch'];steps=plan['steps'];opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True)
 cfg=dict(mode=a.mode,GPU=g,batch=batch,max_steps=steps,seed=125123,sample_seed=125173,initial_sha256=initial['initial_sha256'],parameters=sum(p.numel() for p in net.parameters()),loss='masked conditional tail MSE/100 +0.1 sign BCE; original FIT inclusion weights',oracle_gate=True,official_val_used=False,fit_shuffle_hash=train.perm_hash,monitor_shuffle_hash=monitor.perm_hash,sampling_sha256=hashlib.sha256((A/'sampling.npz').read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'src/c125_net.py']})
 atomic_json(out/'config.json',cfg)
 start=time.monotonic();sampling=np.load(A/'sampling.npz');frozen_ix=torch.tensor(sampling['train'],device='cuda',dtype=torch.long);fitix=torch.tensor(sampling['fit_eval'],device='cuda',dtype=torch.long)
 baseline=evaluate(net,monitor,a.mode,batch);atomic_json(out/'initial.json',baseline);best=float(np.mean(baseline['tail_RMSE']));beststep=0
 torch.save(dict(state=net.state_dict(),step=0,config=cfg),out/'best.pt');history=[]
 for step in range(1,steps+1):
  ix=frozen_ix[step-1]
  lr=.0003*min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
  for pg in opt.param_groups:pg['lr']=lr
  opt.zero_grad(set_to_none=True);v,_=loss(net,train,ix,a.mode);v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();assert torch.isfinite(v)
  if step%250==0 or step==steps:
   val=evaluate(net,monitor,a.mode,batch);criterion=float(np.mean(val['tail_RMSE']))
   if criterion<best:
    best=criterion;beststep=step;torch.save(dict(state=net.state_dict(),step=step,config=cfg),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   rec=dict(step=step,loss=float(v),monitor=val,best_step=beststep,seconds=time.monotonic()-start)
   if step%500==0:rec['fit_panel']=evaluate(net,train,a.mode,batch,indices=fitix)
   history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('STEP',json.dumps(rec),flush=True)
   if step>=1500 and step-beststep>=1000:break
 torch.save(dict(state=net.state_dict(),step=step,config=cfg),out/'last.pt')
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 val=evaluate(net,monitor,a.mode,batch,save=out/'monitor');fit=evaluate(net,train,a.mode,batch,indices=fitix)
 result=dict(status='complete',mode=a.mode,best_step=beststep,last_step=step,monitor=val,fit_panel=fit,seconds=time.monotonic()-start,oracle_gate=True,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--mode',choices=['old','raw','shuffled']);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.smoke:smoke(g)
 else:run(a,g)
if __name__=='__main__':main()
