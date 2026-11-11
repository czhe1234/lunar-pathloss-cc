import sys,time,json,csv,argparse,math,os
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c133_model import ContextTransport as PowerTransport
from src.c134_data import Dataset
import numpy as np,torch
A=R/'audit/c134_paired'
def paired_objective(e,valid,w,pair_ok,groups):
 half=e.shape[1]//2
 mask=valid[:,:half]&valid[:,half:]&pair_ok[:,:,None]
 diff=e[:,:half]-e[:,half:]
 weight=w*mask
 local=(torch.where(mask,diff.square(),0)*w).sum((0,1))/weight.sum((0,1)).clamp_min(1)
 total=torch.zeros(2,device=e.device);count=torch.zeros_like(total)
 for group in groups:
  for j,u in enumerate(group):
   for v in group[j+1:]:
    ok=mask[u]&mask[v];ww=(w[u]+w[v])*.5
    total+=(torch.where(ok,(diff[u]-diff[v]).square(),0)*ww).sum(0)
    count+=(ok*ww).sum(0)
 cross=total/count.clamp_min(1)
 return local.mean()/2/25,cross.mean()/4/25
def paired_check():
 e=torch.zeros(4,8,2,device='cuda');v=torch.ones_like(e,dtype=torch.bool);w=torch.ones(4,1,1,device='cuda');ok=torch.ones(4,4,device='cuda',dtype=torch.bool)
 assert all(float(z)==0 for z in paired_objective(e,v,w,ok,[[0,1,2,3]]))
 e[:,:4]=torch.arange(4,device='cuda')[None,:,None]
 local,cross=paired_objective(e,v,w,ok,[[0,1,2,3]])
 assert local>0 and cross==0
 e=torch.arange(4,device='cuda')[:,None,None].expand(4,8,2).float()
 assert all(float(z)==0 for z in paired_objective(e,v,w,ok,[[0,1,2,3]]))

@torch.no_grad()
def score(net,data,positions=None,save=None):
 net.eval();positions=list(range(len(data.ids))) if positions is None else positions;stat=[];pred=[]
 for st in range(0,len(positions),8):
  ix=positions[st:st+8];g,q,b,y,v=data.batch(ix);p=net(g,q,b);e=torch.where(v,p-y,0.).double();old=torch.where(v,b-y,0.).double();tail=v&((b-y).abs()>=5)
  ss=e.square().sum(1);nn=v.sum(1);ts=torch.where(tail,e.square(),0).sum(1);tn=tail.sum(1)
  stat.append(torch.cat([ss,nn,old.square().sum(1),ts,tn],1).cpu().numpy())
  if save is not None:pred.append(p.cpu().numpy())
 z=np.concatenate(stat);tot=z.sum(0);rm=np.sqrt(tot[:2]/tot[2:4]);base=np.sqrt(tot[4:6]/tot[2:4])
 out=dict(RMSE=rm.tolist(),C82=base.tolist(),gain=(base-rm).tolist(),count=tot[2:4].tolist(),tail_RMSE=np.sqrt(tot[6:8]/np.maximum(tot[8:10],1)).tolist())
 if save is not None:np.save(save,np.concatenate(pred));np.save(str(save).replace('.npy','_stats.npy'),z)
 net.train();return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['plain','paired'],default='paired');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');plan=json.loads((A/'protocol.json').read_text());torch.manual_seed(plan['seed']);start=time.monotonic()
 if a.smoke:
  paired_check()
  # Use completed geometry cases from FIT, no new data selection on labels.
  rows=list(csv.DictReader((DATA/'train_index.csv').open()));ids=[i for i in plan['fit'] if (A/'cache'/rows[i]['terrain_id']/f'{i}.npz').exists()][:16]
  assert len(ids)>=4
  data=Dataset('fit',ids);net=PowerTransport('direction').cuda();data.neutral_cache(net)
  ix=list(range(min(4,len(ids))));g,q,b,y,v=data.batch(ix);p=net(g,q,b)
  initial=float((p-b).abs().max());assert initial<1e-4,initial
  # No source-specific quantities enter K; verify column sums after trainable normalization.
  with torch.no_grad():
   k=net.last_k;mass=torch.zeros(len(ix),2048,2,device='cuda').scatter_add_(1,g['src'].long()[...,None].expand(-1,-1,2),k)
  assert mass.max()<=.85001
  opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001,fused=True);losses=[];times=[]
  for j in range(20):
   torch.cuda.synchronize();t=time.monotonic()
   pixels=torch.randint(data.pp,(len(ix),384),device='cuda');g,q,b,y,v=data.batch(ix,pixels);p=net(g,q,b)
   loss=torch.where(v,(p-y).square(),0).sum()/v.sum()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();assert torch.isfinite(loss)
   torch.cuda.synchronize();losses.append(float(loss));times.append(time.monotonic()-t)
  grads={k:float(p.grad.norm()) for k,p in net.named_parameters() if k.endswith('layers.4.weight')}
  assert all(np.isfinite(x) and x>0 for x in grads.values()),grads
  result=dict(status='passed',GPU=gpu,initial_C82_max_difference=initial,column_mass_max=float(mass.max()),gradient_norms=grads,loss_first_last=[losses[0],losses[-1]],seconds_per_step=float(np.mean(times[5:])),parameters=sum(p.numel() for p in net.parameters()),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  atomic_json(A/'model_smoke.json',result);print('SMOKE',json.dumps(result),flush=True);return
 assert json.loads((A/'model_smoke.json').read_text())['status']=='passed'
 fit=Dataset('fit');mon=Dataset('monitor');net=PowerTransport('direction',plan['rounds']).cuda();fit.neutral_cache(net);mon.neutral_cache(net)
 out=R/'runs'/f'c134_{a.mode}';out.mkdir(exist_ok=False);atomic_json(out/'config.json',dict(plan=plan,GPU=gpu,parameters=sum(p.numel() for p in net.parameters()),initial_sha256=__import__('hashlib').sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in net.state_dict().values())).hexdigest(),official_VAL_used=False))
 opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'],fused=True)
 initial=score(net,mon);atomic_json(out/'initial.json',initial);best=np.mean(initial['RMSE']);be=0;hist=[]
 torch.save(dict(state=net.state_dict(),epoch=0,mode=a.mode,rounds=net.rounds),out/'best.pt')
 groups={}
 for j,t in enumerate(fit.terrain):groups.setdefault(t,[]).append(j)
 keys=list(groups);rng=np.random.default_rng(132073);gen=torch.Generator(device='cuda').manual_seed(132089)
 fitpos=rng.choice(len(fit.ids),min(128,len(fit.ids)),replace=False).tolist()
 steps=0;maxstep=plan['epochs']*math.ceil(len(fit.ids)/16)
 for ep in range(1,plan['epochs']+1):
  group_order=rng.permutation(len(keys));batches=[];batch=[]
  for gi in group_order:
   for j in rng.permutation(groups[keys[gi]]):
    batch.append(int(j))
    if len(batch)==16:batches.append(batch);batch=[]
  if batch:batches.append(batch)
  for ix in batches:
   steps+=1;lr=plan['lr']*min(1.,.2+.8*steps/100)*(.2+.8*.5*(1+math.cos(math.pi*steps/maxstep)))
   for group in opt.param_groups:group['lr']=lr
   pair_indices=torch.randint(fit.pp//2,(1,plan['query_points_per_step']//2),device='cuda',generator=gen)
   pixels=torch.cat([pair_indices,pair_indices+fit.pp//2],1).expand(len(ix),-1)
   g,q,b,y,v=fit.batch(ix,pixels);p=net(g,q,b);w=fit.weights[ix,None,None];e=torch.where(v,p-y,0.)
   # Fixed population denominator: expected loss equals the intended weighted full pixel MSE.
   se=(e.square()*w).sum((0,1));loss=(se/(fit.den*len(ix)*pixels.shape[1]/fit.pp)).mean()/25
   if a.mode=='paired':
    same={}
    for j,ii in enumerate(ix):same.setdefault(fit.terrain[ii],[]).append(j)
    pair_ok=fit.pair_ok[ix][:,pair_indices[0]]
    ll,cl=paired_objective(e,v,w,pair_ok,list(same.values()))
    loss=loss+plan['aux_local']*ll+plan['aux_cross_tx']*cl
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();assert torch.isfinite(loss)
  val=score(net,mon);fv=score(net,fit,fitpos);crit=np.mean(val['RMSE'])
  if crit<best:
   best=crit;be=ep;torch.save(dict(state=net.state_dict(),epoch=ep,mode=a.mode,rounds=net.rounds),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  rec=dict(epoch=ep,steps=steps,monitor=val,fit_panel=fv,best_epoch=be,seconds=time.monotonic()-start,lr=lr)
  hist.append(rec);atomic_json(out/'history.json',hist);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(rec),flush=True)
  torch.save(dict(state=net.state_dict(),epoch=ep,mode=a.mode,rounds=net.rounds),out/'last.pt')
  if ep>=plan['min_epochs'] and ep-be>=plan['patience']:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 val=score(net,mon,save=out/'monitor_sample_prediction.npy');fv=score(net,fit,fitpos)
 result=dict(status='complete',mode=a.mode,best_epoch=be,last_epoch=ep,monitor_sample=val,fit_panel=fv,seconds=time.monotonic()-start,official_VAL_used=False,full_map_validation_pending=True)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
