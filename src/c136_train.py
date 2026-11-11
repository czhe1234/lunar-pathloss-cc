"""Matched native/coarse source-state pilot. Data/model inputs remain isolated."""
import sys,time,json,csv,argparse,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c136_model import NativeState
import numpy as np,torch
A=R/'audit/c136_native'
KEYS=['si','sw','ai','aw','dw','uf','inv','pw','feat','active','txxy','df']
class Dataset:
 def __init__(self,part,ids,device=True):
  self.ids=ids;self.graph=[];self.rows=list(csv.DictReader((DATA/'train_index.csv').open()))
  self.terrain=[self.rows[i]['terrain_id'] for i in ids];self.tids=list(dict.fromkeys(self.terrain));gt={t:j for j,t in enumerate(self.tids)}
  self.gid=[gt[t] for t in self.terrain]
  heights=np.load(DATA/'train_hm.npy',mmap_mode='r');byterrain={r['terrain_id']:int(r['hm_row']) for r in self.rows}
  self.h=torch.as_tensor(np.array(heights[[byterrain[t] for t in self.tids]]),device='cuda')
  src=R/'audit/c101_residual'/part;meta=json.loads((src/'done.json').read_text());self.pp=meta['points_per_map'];ix={i:j for j,i in enumerate(meta['ids'])}
  old=np.load(src/'target.npy',mmap_mode='r').reshape(-1,self.pp,4);base=np.load(src/'base.npy',mmap_mode='r').reshape(-1,self.pp,2);weight=np.load(src/'weight.npy',mmap_mode='r').reshape(-1,self.pp)
  jj=[ix[i] for i in ids];target=torch.as_tensor(np.array(old[jj]),device='cuda')
  self.y=target[:,:,:2];self.v=target[:,:,2:]>0;self.base=torch.as_tensor(np.array(base[jj]),device='cuda')
  w=np.array(weight[jj,0]) if part=='fit' else np.ones(len(ids),np.float32);w=w/w.mean();self.w=torch.as_tensor(w,device='cuda')
  self.den=(self.v*self.w[:,None,None]).sum((0,1))/(len(ids)*self.pp);t=time.monotonic()
  self.cpu_bytes=0
  for j,i in enumerate(ids):
   with np.load(A/'cache'/f'{i}.npz') as ar:
    d={k:np.array(ar[k]) for k in KEYS if k in ar.files}
   nbytes=sum(v.nbytes for v in d.values());self.cpu_bytes+=nbytes
   # Keep enough free device memory for the differentiable native state.
   keep_gpu=torch.cuda.memory_allocated()+nbytes<torch.cuda.get_device_properties(0).total_memory*.70
   self.graph.append({k:torch.as_tensor(v,device='cuda' if keep_gpu else 'cpu') for k,v in d.items()})
   if j%200==0:print('LOAD',part,j,len(ids),'GiB',round(torch.cuda.memory_allocated()/2**30,2),flush=True)
  print('LOADED',part,len(ids),'seconds',round(time.monotonic()-t,2),'data_GiB',self.cpu_bytes/2**30,flush=True)
 def batch(self,ix):
  ug=list(dict.fromkeys(self.gid[i] for i in ix));inverse=[ug.index(self.gid[i]) for i in ix]
  graphs=[{k:(v if v.is_cuda else v.to('cuda')) for k,v in self.graph[i].items()} for i in ix]
  return self.h[ug],graphs,[self.base[i] for i in ix],inverse
@torch.no_grad()
def score(net,data):
 net.eval();ss=np.zeros(2);nn=np.zeros(2);old=np.zeros(2);tailss=np.zeros(2);tailnn=np.zeros(2);sign=np.zeros(2);corr=np.zeros(2);by=[]
 for st in range(0,len(data.ids),4):
  ix=list(range(st,min(st+4,len(data.ids))));h,g,b,inv=data.batch(ix);out=net(h,g,b,inv)
  for j,p in zip(ix,out):
   v=data.v[j];e=torch.where(v,p-data.y[j],0.).double();r=torch.where(v,data.base[j]-data.y[j],0.).double();tail=v&(r.abs()>=5);c=p-data.base[j]
   se=e.square().sum(0).cpu().numpy();n=v.sum(0).cpu().numpy();ss+=se;nn+=n;old+=r.square().sum(0).cpu().numpy()
   tailss+=(e.square()*tail).sum(0).cpu().numpy();tailnn+=tail.sum(0).cpu().numpy()
   sign+=(((c*(data.y[j]-data.base[j]))>0)&tail).sum(0).cpu().numpy();corr+=(c.abs()*tail).sum(0).cpu().numpy()
   by.append([data.ids[j],*se,*n])
 out=dict(RMSE=np.sqrt(ss/nn).tolist(),C82=np.sqrt(old/nn).tolist(),gain=(np.sqrt(old/nn)-np.sqrt(ss/nn)).tolist(),valid_count=nn.tolist(),
  tail_RMSE=np.sqrt(tailss/np.maximum(tailnn,1)).tolist(),tail_sign_accuracy=(sign/np.maximum(tailnn,1)).tolist(),tail_abs_correction=(corr/np.maximum(tailnn,1)).tolist())
 net.train();return out,by
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['native','coarse'],default='native');ap.add_argument('--capacity',action='store_true');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3);torch.set_float32_matmul_precision('high');torch.manual_seed(136041);plan=json.loads((A/'protocol.json').read_text())
 fitids=plan['fit'][:16] if a.capacity else plan['fit']
 if a.smoke:fitids=fitids[:1]
 fit=Dataset('fit',fitids);mon=None if a.capacity or a.smoke else Dataset('monitor',plan['monitor'])
 if a.capacity:fit.w.fill_(1);fit.den=fit.v.float().mean((0,1))
 net=NativeState(a.mode).cuda();sha=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in net.state_dict().values())).hexdigest()
 out=R/'runs'/('c136_capacity16' if a.capacity else f'c136_{a.mode}')
 if not a.smoke:out.mkdir(exist_ok=False);atomic_json(out/'config.json',dict(plan=plan,GPU=gpu,mode=a.mode,capacity=a.capacity,initial_sha256=sha,parameters=sum(p.numel() for p in net.parameters()),fit_maps=len(fitids)))
 opt=torch.optim.AdamW(net.parameters(),lr=.001 if a.capacity else plan['lr'],weight_decay=0 if a.capacity else plan['weight_decay'],fused=True)
 h,g,b,iv=fit.batch(list(range(min(4,len(fitids)))));pred=net(h,g,b,iv)
 init=max(float((x-y).abs().max()) for x,y in zip(pred,b));assert init<.001,init
 updates=0;start=time.monotonic();history=[];best=float('inf');stepsper=math.ceil(len(fitids)/4);total=20 if a.smoke else 2000 if a.capacity else plan['epochs']*stepsper
 rng=np.random.default_rng(136043);order=[];times=[];initial=score(net,fit if mon is None else mon)[0]
 if not a.smoke:atomic_json(out/'initial.json',dict(initial,initial_C82_max_difference=init))
 while updates<total:
  if not order:
   groups={}
   for j,t in enumerate(fit.terrain):groups.setdefault(t,[]).append(j)
   order=[j for k in rng.permutation(len(groups)) for j in list(groups.values())[k]]
  ix=order[:4];order=order[4:];h,g,b,iv=fit.batch(ix)
  torch.cuda.synchronize();ts=time.monotonic()
  lr=(.001 if a.capacity else plan['lr'])*(.1+.9*(1+math.cos(math.pi*updates/total))/2)
  for pg in opt.param_groups:pg['lr']=lr
  pred=torch.stack(net(h,g,b,iv));y=fit.y[ix];v=fit.v[ix];w=fit.w[ix,None,None]
  loss=(((pred-y).square()*v*w).mean((0,1))/fit.den).mean()/25
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  assert torch.isfinite(loss),updates
  torch.cuda.synchronize();times.append(time.monotonic()-ts);updates+=1
  checkpoint=(a.smoke and updates==total) or (a.capacity and (updates%200==0 or updates==total)) or (not a.smoke and not a.capacity and (updates%(10*stepsper)==0 or updates==total))
  if checkpoint:
   result,by=score(net,fit if mon is None else mon);ep=updates/stepsper
   item=dict(updates=updates,epoch=ep,seconds=time.monotonic()-start,train_last_loss=float(loss.detach()),evaluation=result,mean_step_seconds=float(np.mean(times[-100:])))
   history.append(item);print(json.dumps(item),flush=True)
   if not a.smoke:
    torch.save(dict(state=net.state_dict(),updates=updates,mode=a.mode),out/'last.pt');atomic_json(out/'history.json',history)
    if np.mean(result['RMSE'])<best:
     best=np.mean(result['RMSE']);torch.save(dict(state=net.state_dict(),updates=updates,mode=a.mode),out/'best.pt');atomic_json(out/'best.json',item);np.save(out/'best_per_map.npy',np.array(by))
   if a.capacity and max(result['RMSE'])<=1:break
 if a.smoke:
  result=dict(status='passed',GPU=gpu,initial_max_C82_difference=init,first_last_training_loss=[None,float(loss.detach())],step_seconds=float(np.mean(times[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in net.parameters()))
  atomic_json(A/'model_smoke.json',result);print(result,flush=True);return
 done=dict(status='complete',GPU=gpu,mode=a.mode,capacity=a.capacity,updates=updates,seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30,initial_sha256=sha,last=history[-1],scope='fixed sampled TRAIN holdout; no official VAL')
 if not a.capacity:
  # Fit score separates optimization from transfer; it is an unweighted diagnostic.
  done['fit_score']=score(net,fit)[0]
 atomic_json(out/'done.json',done);print('DONE',json.dumps(done),flush=True)
if __name__=='__main__':main()
