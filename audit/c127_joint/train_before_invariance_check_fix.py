import sys,json,time,math,hashlib,argparse,os
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import reader,digest
from src.c99_data import Panel
from src.c99_axial import AxialField
from src.c127_joint import JointField,objective
A=R/'audit/c127_joint'
def create(route):
 torch.manual_seed(127041);net=JointField()
 ck=torch.load(R/'runs/c99_global/best.pt',map_location='cpu',weights_only=False)
 assert ck['epoch']==11
 net.tx.load_state_dict(ck['state'])
 if route=='residual':
  torch.nn.init.zeros_(net.tx.head.weight);torch.nn.init.zeros_(net.tx.head.bias)
 return net
def weights(r,train):
 full=np.load(R/'features/c38_dense/fit_map_indices.npy')
 cnt=Counter(r.rows[int(i)]['terrain_id'] for i in full);sel=Counter(train.terrain)
 w=torch.tensor([cnt[t]/sel[t] for t in train.terrain],device='cuda');return w/w.mean()
def group_batch(data,ix):
 values=ix if isinstance(ix,list) else ix.detach().cpu().tolist()
 groups={}
 for k,j in enumerate(values):groups.setdefault(data.terrain[int(j)],[]).append(k)
 g=list(groups.values());first=torch.tensor([u[0] for u in g],device='cuda')
 inverse=torch.empty(len(values),device='cuda',dtype=torch.long)
 for k,u in enumerate(g):inverse[u]=k
 return g,first,inverse
@torch.inference_mode()
def evaluate(net,data,route,batch=16,positions=None,save=None):
 net.eval();positions=list(range(len(data.ids))) if positions is None else positions
 arr=np.lib.format.open_memmap(save,'w+',dtype=np.float32,shape=(len(positions),2,256,256)) if save else None
 count=torch.zeros(2,device='cuda',dtype=torch.float64);s=count.clone();base_s=count.clone();ts=count.clone();tn=count.clone()
 sc=count.clone();pd=count.clone();band_s=torch.zeros((),device='cuda',dtype=torch.float64);band_n=band_s.clone()
 for st in range(0,len(positions),batch):
  ix=positions[st:st+batch];idx=torch.tensor(ix,device='cuda');x,prior,y,v=data.batch(idx);base=data.reference[idx];ref=base if route=='residual' else prior
  groups,first,inverse=group_batch(data,ix)
  with torch.autocast('cuda',torch.bfloat16):scene,tx=net.components(x,first,inverse)
  p=ref+scene+tx;e=torch.where(v,p-y,0.);eb=torch.where(v,base-y,0.)
  s+=e.double().square().sum((0,2,3));count+=v.sum((0,2,3));base_s+=eb.double().square().sum((0,2,3))
  tail=v&((y-base).abs()>=5);ts+=torch.where(tail,e.double().square(),0).sum((0,2,3));tn+=tail.sum((0,2,3))
  sc+=(tail&((p-base)*(y-base)>0)).sum((0,2,3));pd+=torch.where(tail,(p-base).abs(),0).sum((0,2,3),dtype=torch.float64)
  both=v[:,0]&v[:,1];be=torch.where(both,e[:,1]-e[:,0],0);band_s+=be.double().square().sum();band_n+=both.sum()
  if arr is not None:arr[st:st+len(ix)]=p.float().cpu().numpy()
 if arr is not None:arr.flush()
 net.train()
 return dict(rmse=(s/count).sqrt().tolist(),C82=(base_s/count).sqrt().tolist(),sse=s.tolist(),count=count.tolist(),
  tail_rmse=(ts/tn).sqrt().tolist(),tail_count=tn.tolist(),tail_sign_accuracy=(sc/tn).tolist(),tail_mean_abs_correction=(pd/tn).tolist(),
  band_difference_rmse=float((band_s/band_n).sqrt()))
def smoke(gpu):
 plan=json.loads((A/'protocol.json').read_text());r=reader()
 group=[u for u in plan['groups'].values() if len(u)==4][:4];ids=[plan['fit'][j] for u in group for j in u]
 train=Panel(r,ids,reference=True);w=torch.ones(len(ids),device='cuda');ix=list(range(16));groups,first,inverse=group_batch(train,ix)
 x,prior,y,v=train.batch(torch.arange(16,device='cuda'))
 snapshots={};timing=[];initial_error={}
 for route in ['direct','residual']:
  net=create(route);initial=digest(net);torch.save(dict(state=net.state_dict(),initial_sha256=initial),A/f'initial_{route}.pt')
  net=net.cuda().to(memory_format=torch.channels_last);ref=train.reference if route=='residual' else prior
  with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
   s,t=net.components(x,first,inverse);allp=ref+s+t
   singles=torch.cat([net(x[j:j+1]) for j in range(16)])
  invariant=float((singles-(s+t)).abs().max());assert invariant<.08
  if route=='residual':assert float((s+t).abs().max())==0.
  else:
   old=AxialField(True).cuda();old.load_state_dict(torch.load(R/'runs/c99_global/best.pt',map_location='cpu',weights_only=False)['state'])
   with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):oldp=prior+old(x).float()
   mismatch=(allp-oldp).float();initial_error[route]=dict(rms=float(mismatch.square().mean().sqrt()),max=float(mismatch.abs().max()))
   assert initial_error[route]['rms']<.03;del old
  # Identical terrain inputs yield identical S; no group labels enter prediction.
  assert torch.allclose(s[:4],s[:1].expand_as(s[:4]))
  opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'],fused=True)
  hist=[]
  for step in range(12):
   torch.cuda.synchronize();start=time.monotonic()
   with torch.autocast('cuda',torch.bfloat16):s,t=net.components(x,first,inverse)
   loss,parts,ss,cc=objective(s,t,ref,y,v,w,groups,'joint')
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
   assert torch.isfinite(loss)
   torch.cuda.synchronize();timing.append(time.monotonic()-start);hist.append(float(loss))
  grads={k:float(p.grad.float().square().mean().sqrt()) for k,p in net.named_parameters() if k in ['tx.stem.weight','tx.head.weight','scene.stem.weight','scene.head.weight']}
  assert all(np.isfinite(z) and z>0 for z in grads.values())
  # Supervision algebra: correct mean/deviation should have zero component and band errors.
  known_scene=torch.zeros_like(y);known_tx=torch.zeros_like(y)
  for u in groups:
   vv=v[u];num=vv.sum(0);mu=torch.where(vv,y[u]-ref[u],0).sum(0)/num.clamp_min(1)
   known_scene[u]=mu;known_tx[u]=y[u]-ref[u]-mu
  truth_loss,*_=objective(known_scene,known_tx,ref,y,v,w,groups,'joint');assert float(truth_loss)<1e-9
  snapshots[route]=dict(initial_sha256=initial,parameters=sum(p.numel() for p in net.parameters()),gradient_rms=grads,batch_invariance_max=invariant,
   loss_first_last=[hist[0],hist[-1]],truth_decomposition_loss=float(truth_loss))
  del net,opt;torch.cuda.empty_cache()
 out=dict(status='passed',GPU=gpu,arms=snapshots,initial_direct_replay_error=initial_error,seconds_per_step=float(np.mean(timing[-8:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30,official_val_used=False)
 atomic_json(A/'smoke.json',out);print('SMOKE',json.dumps(out),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--route',choices=['direct','residual']);ap.add_argument('--mode',choices=['independent','joint']);args=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 if args.smoke:smoke(gpu);return
 assert json.loads((A/'smoke.json').read_text())['status']=='passed'
 plan=json.loads((A/'protocol.json').read_text());r=reader();started=time.monotonic()
 train=Panel(r,plan['fit'],reference=True);monitor=Panel(r,plan['monitor'],reference=True);w=weights(r,train)
 net=create(args.route);initial=torch.load(A/f'initial_{args.route}.pt',map_location='cpu',weights_only=False);net.load_state_dict(initial['state']);assert digest(net)==initial['initial_sha256']
 net=net.cuda().to(memory_format=torch.channels_last)
 out=R/f'runs/c127_{args.route}_{args.mode}';out.mkdir(exist_ok=False)
 config=dict(route=args.route,mode=args.mode,GPU=gpu,batch=plan['batch'],max_epochs=plan['max_epochs'],lr=plan['lr'],weight_decay=plan['weight_decay'],initial_sha256=initial['initial_sha256'],
  protocol_sha256=hashlib.sha256((A/'protocol.json').read_bytes()).hexdigest(),source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'src/c127_joint.py']},
  parameters=sum(p.numel() for p in net.parameters()),input=plan['input'],FIT=len(train.ids),monitor=len(monitor.ids),official_val_used=False)
 atomic_json(out/'config.json',config);atomic_json(out/'indices.json',monitor.ids)
 opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'],fused=True)
 initial_metric=evaluate(net,monitor,args.route,plan['batch']);atomic_json(out/'initial.json',initial_metric)
 baseline=np.array(initial_metric['C82']);best=np.mean(initial_metric['rmse']);be=0;hist=[]
 torch.save(dict(state=net.state_dict(),epoch=0,config=config),out/'best.pt')
 fit=evaluate(net,train,args.route,plan['batch'],plan['fit_eval_positions']);atomic_json(out/'fit_initial.json',fit)
 print('INITIAL',json.dumps(dict(route=args.route,mode=args.mode,monitor=initial_metric['rmse'],FIT_panel=fit['rmse'],parameters=config['parameters'])),flush=True)
 total_steps=sum(map(len,plan['orders']));global_step=0
 for ep,batches in enumerate(plan['orders'],1):
  total=torch.zeros(4,device='cuda',dtype=torch.float64);parts_sum=torch.zeros(4,device='cuda')
  for j,ix in enumerate(batches):
   global_step+=1
   lr=plan['lr']*min(1.,.1+.9*global_step/100)*(.15+.85*.5*(1+math.cos(math.pi*(global_step-1)/total_steps)))
   for pg in opt.param_groups:pg['lr']=lr
   idx=torch.tensor(ix,device='cuda');x,prior,y,v=train.batch(idx);ref=train.reference[idx] if args.route=='residual' else prior
   groups,first,inverse=group_batch(train,ix)
   with torch.autocast('cuda',torch.bfloat16):s,t=net.components(x,first,inverse)
   loss,parts,ss,cc=objective(s,t,ref,y,v,w[idx],groups,args.mode)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
   assert torch.isfinite(loss),f'nonfinite epoch {ep} step {j}'
   total+=torch.cat([ss.detach(),cc]);parts_sum+=torch.stack([parts[k].detach() for k in ['risk','scene','tx','band']])
   if (j+1)%80==0:
    rec=dict(epoch=ep,step=j+1,steps=len(batches),fit_running_rmse=(total[:2]/total[2:]).sqrt().tolist(),seconds=time.monotonic()-started)
    atomic_json(out/'progress.json',rec);print('PROGRESS',json.dumps(rec),flush=True)
  val=evaluate(net,monitor,args.route,plan['batch']);criterion=np.mean(val['rmse'])
  if criterion<best:
   best=criterion;be=ep;torch.save(dict(state=net.state_dict(),epoch=ep,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  fit=evaluate(net,train,args.route,plan['batch'],plan['fit_eval_positions'])
  rec=dict(epoch=ep,fit_running_rmse=(total[:2]/total[2:]).sqrt().tolist(),fit_panel=fit,loss_parts=(parts_sum/len(batches)).tolist(),monitor=val,
   gain=(baseline-val['rmse']).tolist(),best_epoch=be,lr=lr,seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  hist.append(rec);atomic_json(out/'history.json',hist);atomic_json(out/'progress.json',rec)
  torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),epoch=ep,config=config),out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
  print('EPOCH',json.dumps({k:v for k,v in rec.items() if k not in ['fit_panel','monitor']}|dict(fit_panel=fit['rmse'],monitor=val['rmse'])),flush=True)
  if ep>=plan['min_epochs'] and ep-be>=plan['patience']:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 val=evaluate(net,monitor,args.route,plan['batch'],save=out/'monitor_prediction.npy')
 fit=evaluate(net,train,args.route,plan['batch'],plan['fit_eval_positions'],save=out/'fit_prediction.npy')
 atomic_json(out/'fit_ids.json',[train.ids[j] for j in plan['fit_eval_positions']])
 result=dict(status='complete',route=args.route,mode=args.mode,best_epoch=be,last_epoch=ep,monitor=val,FIT_panel=fit,gain=(baseline-val['rmse']).tolist(),
  seconds=time.monotonic()-started,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
