import sys,json,time,math,hashlib,argparse,os
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c105_context import reader,ResidualPanel,digest
from src.c126_statefield import StateField,losses,evaluate
import torch,numpy as np
A=R/'audit/c126_statefield'
def weights(r,train):
 full=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(r.rows[int(i)]['terrain_id'] for i in full);sel=Counter(train.terrain)
 w=torch.tensor([cnt[t]/sel[t] for t in train.terrain],device='cuda');return w/w.mean()
def prepare(g):
 old=json.load(open(R/'audit/c105_context/protocol.json'));r=reader();train=ResidualPanel(r,old['fit'],'context');weight=weights(r,train)
 count=torch.zeros(2,3,device='cuda',dtype=torch.float64);total=count.clone()
 for st in range(0,len(train.ids),64):
  sl=slice(st,st+64);res=train.y[sl]-train.reference[sl];v=train.w[sl];cls=torch.where(res>=5,1,torch.where(res<=-5,2,0));w=weight[sl,None,None,None]
  for k in range(3):
   m=v&(cls==k);count[:,k]+=(m*w).sum((0,2,3),dtype=torch.float64);total[:,k]+=(torch.where(m,res,0)*w).sum((0,2,3),dtype=torch.float64)
 prior=(count/count.sum(1,keepdim=True)).float().cpu();means=(total/count).float().cpu()
 torch.manual_seed(126041);net=StateField(prior,means);ih=digest(net);initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};torch.save(dict(state=initial,initial_sha256=ih),A/'initial.pt')
 net=net.cuda().to(memory_format=torch.channels_last);ix=torch.arange(8,device='cuda');x,b,y,v=train.batch(ix)
 with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):delta,*_=net.outputs(x)
 maximum=float(delta.abs().max());assert maximum<1e-5
 opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True);tim=[]
 for step in range(8):
  torch.cuda.synchronize();st=time.monotonic()
  with torch.autocast('cuda',torch.bfloat16):d,l,m,p=net.outputs(x)
  value,parts,ss,cc=losses(d,l,m,b,y,v,weight[ix],net.prior,'state')
  opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(value);torch.cuda.synchronize();tim.append(time.monotonic()-st)
 gradients={k:float(v.grad.float().square().mean().sqrt()) for k,v in net.named_parameters() if k in ['stem.weight','head.weight','local.0.weight']}
 assert all(z>0 and np.isfinite(z) for z in gradients.values())
 orders=np.stack([np.random.default_rng(126073+ep).permutation(len(train.ids)) for ep in range(8)]).astype('int32');np.save(A/'orders.npy',orders)
 plan=dict(batch=8,epochs=8,prior=prior.tolist(),means=means.tolist(),initial_sha256=ih,parameters=sum(p.numel() for p in net.parameters()),order_sha256=hashlib.sha256((A/'orders.npy').read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'src/c126_statefield.py']},official_val_used=False)
 atomic_json(A/'protocol.json',plan);atomic_json(A/'smoke.json',dict(GPU=g,maximum_initial_correction=maximum,gradients=gradients,seconds_per_step=float(np.mean(tim[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30));print('PREPARED',plan['parameters'],prior.tolist(),means.tolist(),maximum,tim,flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--mode',choices=['risk','state']);a=ap.parse_args();g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 if a.prepare:prepare(g);return
 plan=json.load(open(A/'protocol.json'));old=json.load(open(R/'audit/c105_context/protocol.json'));batch=plan['batch'];r=reader();start=time.monotonic()
 train=ResidualPanel(r,old['fit'],'context');monitor=ResidualPanel(r,old['monitor'],'context');weight=weights(r,train)
 net=StateField(torch.tensor(plan['prior']),torch.tensor(plan['means']));ck=torch.load(A/'initial.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);assert digest(net)==plan['initial_sha256'];net=net.cuda().to(memory_format=torch.channels_last)
 out=R/f'runs/c126_{a.mode}';out.mkdir(exist_ok=False);atomic_json(out/'indices.json',monitor.ids)
 cfg=dict(mode=a.mode,GPU=g,initial_sha256=plan['initial_sha256'],order_sha256=plan['order_sha256'],batch=batch,epochs=plan['epochs'],FIT=len(train.ids),monitor=len(monitor.ids),aux_weight=.25 if a.mode=='state' else 0.,official_val_used=False,parameters=plan['parameters'])
 atomic_json(out/'config.json',cfg);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True);orders=torch.tensor(np.load(A/'orders.npy'),device='cuda',dtype=torch.long)
 initial=evaluate(net,monitor,batch,force_base=True);atomic_json(out/'initial.json',initial);be=0;best=float(np.mean(initial['rmse']));history=[];nstep=math.ceil(len(train.ids)/batch)
 torch.save(dict(state=net.state_dict(),epoch=0,config=cfg,kind='C82'),out/'best.pt')
 for ep in range(1,plan['epochs']+1):
  net.train();ss=torch.zeros(4,device='cuda',dtype=torch.float64);parts_sum=torch.zeros(3,device='cuda');batches=0
  for j,ix in enumerate(orders[ep-1].split(batch)):
   step=(ep-1)*nstep+j+1;lr=.0003*min(1.,.1+.9*step/100)*(.15+.85*.5*(1+math.cos(math.pi*(step-1)/(nstep*plan['epochs']))))
   for pg in opt.param_groups:pg['lr']=lr
   x,b,y,v=train.batch(ix)
   with torch.autocast('cuda',torch.bfloat16):d,l,mu,p=net.outputs(x)
   value,parts,s,c=losses(d,l,mu,b,y,v,weight[ix],net.prior,a.mode)
   opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(value)
   ss+=torch.cat([s.detach(),c]);parts_sum+=torch.stack([parts[k].detach() for k in ['risk','classCE','amplitude']]);batches+=1
  val=evaluate(net,monitor,batch);criterion=float(np.mean(val['rmse']))
  if criterion<best:
   best=criterion;be=ep;torch.save(dict(state=net.state_dict(),epoch=ep,config=cfg,kind='model'),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  rec=dict(epoch=ep,fit_running_rmse=(ss[:2]/ss[2:]).sqrt().tolist(),loss_parts=(parts_sum/batches).tolist(),monitor=val,gain=(np.array(initial['rmse'])-val['rmse']).tolist(),best_epoch=be,seconds=time.monotonic()-start)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(rec),flush=True)
  if ep>=4 and ep-be>=3:break
 torch.save(dict(state=net.state_dict(),epoch=ep,config=cfg,kind='model'),out/'last.pt')
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);val=evaluate(net,monitor,batch,force_base=be==0,save=out/'monitor_prediction.npy')
 result=dict(status='complete',mode=a.mode,best_epoch=be,last_epoch=ep,monitor=val,gain=(np.array(initial['rmse'])-val['rmse']).tolist(),seconds=time.monotonic()-start,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
