"""Bounded 512-terrain pilot. Point control first, same data/order/budget for spatial model."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
from pathlib import Path
import sys,json,time,math,shutil,hashlib
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard,check_source_unchanged
from c17_spatial_residual import Model
torch.set_num_threads(8);torch.set_float32_matmul_precision('high');info=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15)
check_source_unchanged()
D=R/'features/c17_spatial';meta=json.loads((D/'metadata.json').read_text());assert meta['state']=='completed'
O=R/'runs/c17_spatial_residual_s20261101';O.mkdir(exist_ok=False);(O/'source').mkdir()
for p in [Path(__file__),R/'src/c17_spatial_residual.py',R/'audit/c17_protocol.json',D/'metadata.json']:shutil.copy2(p,O/'source'/p.name)
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
data={}
for part in ['fit','dev']:
 ids=np.array(meta['ids'][part]);x=np.load(D/f'{part}_x.npy',mmap_mode='r');base=np.load(D/f'{part}_base.npy',mmap_mode='r')
 data[part]={'x':torch.from_numpy(np.array(x)).cuda(),'base':torch.from_numpy(np.array(base)).cuda(),
 'y':torch.from_numpy(np.stack([v[ids] for v in Y],1)).cuda(),
 'mask':torch.from_numpy(np.stack([np.unpackbits(v[ids],axis=1,bitorder='big').reshape(-1,256,256) for v in M],1)).cuda()}
probe=data['fit']['x'][:,:,::8,::8].float();mean=probe.mean((0,2,3));std=probe.std((0,2,3)).clamp_min(.05);del probe
@torch.inference_mode()
def evaluate(model,part,save=False):
 d=data[part];sse=torch.zeros(2,device='cuda',dtype=torch.float64);count=torch.zeros_like(sse);pr=[]
 if model is not None:model.eval()
 for lo in range(0,len(d['x']),8):
  sl=slice(lo,lo+8);p=d['base'][sl]
  if model is not None:
   with torch.autocast('cuda',dtype=torch.bfloat16):p=p+model(d['x'][sl])
  err=(p.double()-d['y'][sl].double());m=d['mask'][sl];sse+=(err.square()*m).sum((0,2,3));count+=m.sum((0,2,3)).double()
  if save:pr.append(p.cpu().numpy())
 return {'rmse':(sse/count).sqrt().cpu().tolist(),'valid_pixels':count.cpu().tolist()},np.concatenate(pr) if save else None
baseline={part:evaluate(None,part)[0] for part in ['fit','dev']}
(O/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
print(json.dumps({'event':'loaded','baseline':baseline,'GPU_allocated_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
results={};start=time.time()
for kind in ['point','spatial']:
 torch.manual_seed(20261101);rng=np.random.default_rng(20261101)
 model=Model(kind,mean,std).cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
 run=O/kind;run.mkdir();hist=[];best=float('inf');best_epoch=0
 config={'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),'batch':8,'max_epochs':30,'lr':.0005,'final_lr':.00002,'weight_decay':.0001,'GPU':info,'seed':20261101,'normalization':'FIT-only channel moments, sampled every8pixels, std floor.05','loss':'equal-band valid-pixel residual MSE /25; masks only in loss','selection':'minimum mean band MSE on 128-terrain DEV'}
 (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 initial,_=evaluate(model,'dev');assert np.max(abs(np.array(initial['rmse'])-np.array(baseline['dev']['rmse'])))<1e-10
 for epoch in range(1,31):
  model.train();order=rng.permutation(len(data['fit']['x']));lr=.00002+.5*(.0005-.00002)*(1+math.cos(math.pi*(epoch-1)/30))
  for g in opt.param_groups:g['lr']=lr
  for lo in range(0,len(order),8):
   j=torch.as_tensor(order[lo:lo+8],device='cuda');d=data['fit'];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):res=model(d['x'][j])
   e=(d['base'][j]+res-d['y'][j].float())/5;m=d['mask'][j]
   loss=((e.square()*m).sum((0,2,3))/m.sum((0,2,3)).clamp_min(1)).mean()
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
  dev,_=evaluate(model,'dev')
  rec={'model':kind,'epoch':epoch,'dev':dev,'lr':lr,'seconds':time.time()-start,'GPU_peak_GB':torch.cuda.max_memory_allocated()/1e9}
  if epoch%5==0:rec['fit']=evaluate(model,'fit')[0]
  hist.append(rec);(run/'history.json').write_text(json.dumps(hist,indent=2)+'\n');(O/'progress.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
  metric=float(np.mean(np.array(dev['rmse'])**2))
  if metric<best:
   best=metric;best_epoch=epoch;torch.save({'model':model.state_dict(),'config':config,'epoch':epoch},run/'best.pt')
  if epoch>=10 and epoch-best_epoch>=8:break
 state=torch.load(run/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(state['model'])
 dev,pred=evaluate(model,'dev',save=True);fit,_=evaluate(model,'fit');np.save(run/'dev_prediction.npy',pred)
 indep=[]
 for k in [0,1]:
  m=data['dev']['mask'][:,k].cpu().numpy().astype(bool);truth=data['dev']['y'][:,k].cpu().numpy()
  err=pred[:,k][m].astype(float)-truth[m].astype(float);indep.append(float(np.sqrt(np.mean(err*err))))
 assert np.max(abs(np.array(indep)-np.array(dev['rmse'])))<1e-8
 gains=(100*(1-np.array(indep)/baseline['dev']['rmse'])).tolist()
 results[kind]={'dev_RMSE':indep,'fit_RMSE':fit['rmse'],'relative_RMSE_reduction_percent':gains,'best_epoch':best_epoch,'last_epoch':epoch,'valid_DEV_pixels':dev['valid_pixels'],'weights_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()}
 (run/'result.json').write_text(json.dumps(results[kind],indent=2)+'\n');print(json.dumps({'model_result':kind,**results[kind]}),flush=True)
 del model,opt;torch.cuda.empty_cache()
g=np.array(results['spatial']['relative_RMSE_reduction_percent']);c=np.array(results['point']['relative_RMSE_reduction_percent'])
final={'state':'completed','scope':'small terrain-disjoint dense internalDEV pilot only; not official score','baseline':baseline,'results':results,'promote':bool(np.all(g>=10)&np.all(g-c>=3)),'seconds':time.time()-start,'peak_GPU_GB':torch.cuda.max_memory_allocated()/1e9,'source_unchanged':True}
check_source_unchanged();(O/'result.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
