"""Matched-budget learning control; TRAIN-only. Full-map confirmation is separate."""
import sys,time,json,csv,argparse,math,os
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c132_model import PowerTransport
import numpy as np,torch
A=R/'audit/c132_transport'
FKEYS=['sf','qf','af','df']
def tensor(x):return torch.as_tensor(np.array(x,copy=True),device='cuda')
class Dataset:
 def __init__(self,part,ids=None):
  src=R/'audit/c101_residual'/part;meta=json.loads((src/'done.json').read_text());self.ids=meta['ids'] if ids is None else ids;self.pp=meta['points_per_map']
  self.rows=list(csv.DictReader((DATA/'train_index.csv').open()));self.terrain=[self.rows[i]['terrain_id'] for i in self.ids];self.tids=sorted(set(self.terrain));self.glookup={t:i for i,t in enumerate(self.tids)};self.gid=torch.tensor([self.glookup[t] for t in self.terrain],device='cuda')
  self.g={};self.q={};started=time.monotonic()
  for j,t in enumerate(self.tids):
   ar=np.load(A/'cache'/t/'graph.npz')
   for k in ['src','dst','k','kf','area']:
    x=ar[k]
    if k not in self.g:self.g[k]=torch.empty((len(self.tids),*x.shape),dtype=tensor(x[:1]).dtype,device='cuda')
    self.g[k][j]=tensor(x)
  oldlookup={i:j for j,i in enumerate(meta['ids'])}
  yy=np.load(src/'target.npy',mmap_mode='r').reshape(-1,self.pp,4);base=np.load(src/'base.npy',mmap_mode='r').reshape(-1,self.pp,2)
  self.y=torch.empty(len(self.ids),self.pp,2,device='cuda');self.v=torch.empty_like(self.y,dtype=torch.bool);self.base=torch.empty_like(self.y)
  for j,i in enumerate(self.ids):
   ar=np.load(A/'cache'/self.terrain[j]/f'{i}.npz')
   for k in ['sq','sf','ri','aw','rq','qf','af','df','ds']:
    x=ar[k]
    if k not in self.q:self.q[k]=torch.empty((len(self.ids),*x.shape),dtype=tensor(x[:1]).dtype,device='cuda')
    self.q[k][j]=tensor(x)
   idx=oldlookup[i];self.y[j]=tensor(yy[idx,:,:2]);self.v[j]=tensor(yy[idx,:,2:]).bool();self.base[j]=tensor(base[idx])
   if j%400==0:print('LOAD',part,j,len(self.ids),round(time.monotonic()-started,1),flush=True)
  full=np.load(R/'features/c38_dense/fit_map_indices.npy');counts=Counter(self.rows[int(i)]['terrain_id'] for i in full);sel=Counter(self.terrain)
  self.weights=torch.tensor([counts[t]/sel[t] for t in self.terrain],device='cuda',dtype=torch.float32) if part=='fit' else torch.ones(len(self.ids),device='cuda')
  self.weights/=self.weights.mean()
  self.den=(self.v*self.weights[:,None,None]).sum((0,1))/len(self.ids)
  print('LOADED',part,len(self.ids),len(self.tids),round(time.monotonic()-started,1),'GiB',torch.cuda.memory_allocated()/2**30,flush=True)
 def batch(self,ix,pixels=None):
  ix=torch.as_tensor(ix,device='cuda',dtype=torch.long);g={k:v[self.gid[ix]] for k,v in self.g.items()};q={}
  for k,v in self.q.items():
   if k in ['sq','sf'] or pixels is None:q[k]=v[ix]
   else:q[k]=v[ix[:,None],pixels]
  y=self.y[ix] if pixels is None else self.y[ix[:,None],pixels];v=self.v[ix] if pixels is None else self.v[ix[:,None],pixels];base=self.base[ix] if pixels is None else self.base[ix[:,None],pixels]
  return g,q,base,y,v
 def neutral_cache(self,net):
  z=torch.empty(len(self.ids),self.pp,2,device='cuda')
  with torch.no_grad():
   for st in range(0,len(self.ids),4):
    g,q,*_=self.batch(list(range(st,min(st+4,len(self.ids)))))
    z[st:st+len(q['sq'])]=net.physical(g,q,neutral=True)
  self.q['g0']=z
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
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['direction','pooled','local'],default='direction');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');plan=json.loads((A/'protocol.json').read_text());torch.manual_seed(plan['seed']);start=time.monotonic()
 if a.smoke:
  # Use completed geometry cases from FIT, no new data selection on labels.
  rows=list(csv.DictReader((DATA/'train_index.csv').open()));ids=[i for i in plan['fit'] if (A/'cache'/rows[i]['terrain_id']/f'{i}.npz').exists()][:16]
  assert len(ids)>=4
  data=Dataset('fit',ids);net=PowerTransport(a.mode).cuda();data.neutral_cache(net)
  ix=list(range(min(4,len(ids))));g,q,b,y,v=data.batch(ix);p=net(g,q,b)
  initial=float((p-b).abs().max());assert initial<1e-4,initial
  # No source-specific quantities enter K; verify column sums after trainable normalization.
  with torch.no_grad():
   k=net.kernels(g);mass=torch.zeros(len(ix),2048,2,device='cuda').scatter_add_(1,g['src'].long()[...,None].expand(-1,-1,2),k)
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
 fit=Dataset('fit');mon=Dataset('monitor');net=PowerTransport('direction' if a.mode=='local' else a.mode,0 if a.mode=='local' else plan['rounds']).cuda();fit.neutral_cache(net);mon.neutral_cache(net)
 out=R/'runs'/f'c132_{a.mode}';out.mkdir(exist_ok=False);atomic_json(out/'config.json',dict(plan=plan,GPU=gpu,parameters=sum(p.numel() for p in net.parameters()),initial_sha256=__import__('hashlib').sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in net.state_dict().values())).hexdigest(),official_VAL_used=False))
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
   pixels=torch.randint(fit.pp,(len(ix),plan['query_points_per_step']),device='cuda',generator=gen)
   g,q,b,y,v=fit.batch(ix,pixels);p=net(g,q,b);w=fit.weights[ix,None,None];e=torch.where(v,p-y,0.)
   # Fixed population denominator: expected loss equals the intended weighted full pixel MSE.
   se=(e.square()*w).sum((0,1));loss=(se/(fit.den*len(ix)*pixels.shape[1]/fit.pp)).mean()/25
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
