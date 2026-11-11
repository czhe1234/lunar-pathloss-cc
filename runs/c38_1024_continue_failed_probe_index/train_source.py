"""C38/C39: matched continuation from C37,1024vs256points/map."""
import sys,json,time,math,argparse,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np
import torch
from src.c34_model import Model
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['ple','periodic'],required=True)
p.add_argument('--members',type=int,default=8)
p.add_argument('--width',type=int,default=512)
p.add_argument('--steps',type=int,default=20000)
p.add_argument('--batch',type=int,default=4096)
p.add_argument('--name',required=True)
p.add_argument('--smoke',action='store_true')
p.add_argument('--dense',action='store_true')
p.add_argument('--points',type=int,choices=[256,1024],default=1024)
p.add_argument('--parent',default='c37_wide_periodic8')
a=p.parse_args();a.dense=True
parent=R/'runs'/a.parent
parentcfg=json.loads((parent/'config.json').read_text());parentresult=json.loads((parent/'result.json').read_text())
a.mode=parentcfg['mode'];a.width=parentcfg['width'];a.members=parentcfg['members']
gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.45 if a.points==1024 else .15);torch.set_num_threads(8)
torch.manual_seed(20261205);np.random.seed(20261205)
torch.set_float32_matmul_precision('high')
o=R/'runs'/a.name;o.mkdir(exist_ok=False)
shutil.copy2(Path(__file__),o/'train_source.py');shutil.copy2(R/'src/c34_model.py',o/'model_source.py')
data={}
for part in ['fit','dev']:
 xd=('c38_dense' if a.points==1024 else 'c35_dense') if part=='fit' else 'c10_points'
 yd=('c38_dense' if a.points==1024 else 'c35_dense') if part=='fit' else 'c6_points'
 x=np.load(R/f'features/{xd}/{part}_x.npy',mmap_mode='r')
 d={name:np.load(R/f'features/{yd}/{part}_{name}.npy',mmap_mode='r') for name in ['y','fspl','mask']}
 if a.smoke: x=x[:32768];d={k:v[:32768] for k,v in d.items()}
 data[part]=dict(x=torch.from_numpy(np.array(x)).cuda(),y=torch.tensor((d['y']-d['fspl'])/20,device='cuda'),mask=torch.tensor(np.array(d['mask'],np.float32),device='cuda'),fspl=np.array(d['fspl']),raw=np.array(d['y']))
nm=np.load(R/'audit/c34_numeric_network_20261101/normalization.npz')
mean,std,edges=[torch.tensor(nm[k],device='cuda') for k in ['mean','std','edges']]
target_mean=torch.tensor(nm['target_mean'],device='cuda')
model=Model(a.mode,a.members,a.width,mean,std,edges,target_mean).cuda()
parentck=torch.load(parent/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(parentck['model'])
opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=1e-4)
gen=torch.Generator(device='cuda').manual_seed(20261205)
config={**vars(a),'gpu':gpu,'seed':20261205,'features':294,'continuation':True,'memory_fraction_cap':(.45 if a.points==1024 else .15),'parent_step':parentck['step'],'initial_lr':.0001,'train_rows':len(data['fit']['x']),'dev_rows':len(data['dev']['x']),'parameters':sum(v.numel() for v in model.parameters()),'targets':'(PL-FSPL)/20, separate valid mask per band','normalization':'FIT only','official_VAL_used':False,'architecture':'Independent implementation of numerical embeddings and BatchEnsemble-style MLP; 3 hidden layers','selection':'mean of band RMSE on same sampled DEV as C10; patience6000 steps; no official validation selection'}
atomic_json(o/'config.json',config)
@torch.inference_mode()
def evaluate(part,ids=None,save=False):
 model.eval();d=data[part]
 if ids is None:ids=torch.arange(len(d['x']),device='cuda')
 s=torch.zeros(2,device='cuda',dtype=torch.float64);c=torch.zeros_like(s);pred=[]
 for lo in range(0,len(ids),4096):
  j=ids[lo:lo+4096]
  with torch.autocast('cuda',dtype=torch.bfloat16):q=model(d['x'][j]).mean(1)
  e=(q-d['y'][j]).double()*20;m=d['mask'][j]
  s+=(e.square()*m).sum(0);c+=m.sum(0).double()
  if save:pred.append(q.cpu().numpy()*20+d['fspl'][j.cpu().numpy()])
 model.train()
 return {'rmse':(s/c).sqrt().cpu().tolist(),'sse':s.cpu().tolist(),'count':c.cpu().tolist()},np.concatenate(pred) if save else None
trainprobe=torch.linspace(0,len(data['fit']['x'])-1,min(131072,len(data['fit']['x'])),device='cuda').long()
best=float(np.mean(parentresult['DEV_RMSE']));beststep=0;hist=[];t0=time.time();step_times=[]
torch.save({'model':model.state_dict(),'step':0,'config':config},o/'best.pt')
for step in range(1,a.steps+1):
 ts=time.perf_counter();j=torch.randint(len(data['fit']['x']),(a.batch,),device='cuda',generator=gen)
 warm=min(1.,step/200);lr=warm*(.00001+.5*(.0001-.00001)*(1+math.cos(math.pi*(step-1)/a.steps)))
 for g in opt.param_groups:g['lr']=lr
 opt.zero_grad(set_to_none=True);d=data['fit']
 with torch.autocast('cuda',dtype=torch.bfloat16):q=model(d['x'][j])
 m=d['mask'][j,None,:];loss=((q-d['y'][j,None,:]).square()*m).sum((0,1))/(m.sum((0,1))*a.members).clamp_min(1)
 loss.mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
 if step<=30:
  torch.cuda.synchronize();step_times.append(time.perf_counter()-ts)
 if step==30:
  print(json.dumps({'event':'throughput','samples_per_second':a.batch/np.mean(step_times[10:]),'step_seconds':float(np.mean(step_times[10:])),'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
 if step%1000==0 or step==a.steps:
  val,_=evaluate('dev');fit,_=evaluate('fit',trainprobe);score=np.mean(val['rmse'])
  rec={'step':step,'DEV':val,'FIT_probe':fit,'lr':lr,'seconds':time.time()-t0}
  if score<best:
   best=float(score);beststep=step
   torch.save({'model':model.state_dict(),'step':step,'config':config},o/'best.pt')
  rec['best_step']=beststep;hist.append(rec)
  atomic_json(o/'progress.json',rec);atomic_json(o/'history.json',hist);print(json.dumps(rec),flush=True)
  if not a.smoke and step-beststep>=6000:break
ck=torch.load(o/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
val,pred=evaluate('dev',save=True);fit,_=evaluate('fit')
np.save(o/'dev_prediction.npy',pred)
ind=[]
for k in [0,1]:
 m=data['dev']['mask'][:,k].cpu().numpy().astype(bool);e=pred[m,k].astype(float)-data['dev']['raw'][m,k].astype(float)
 ind.append(float(np.sqrt(np.mean(e*e))))
assert np.max(np.abs(np.array(ind)-val['rmse']))<1e-5
ref=np.array([2.948508059683078,3.387646886271431]);gain=1-np.array(ind)/ref
result={'status':'completed','DEV_RMSE':ind,'FIT_RMSE':fit['rmse'],'best_step':beststep,'last_step':step,'relative_gain_vs_C10':gain.tolist(),'full_DEV_gate':bool((gain>=.03).all()),'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'seconds':time.time()-t0,'smoke':a.smoke}
atomic_json(o/'result.json',result);print(json.dumps(result),flush=True)
