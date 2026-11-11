"""One bounded continuous-regression challenger on the same C10 information.
No new physical input, no official VAL. GPU3 only. Auxiliary sigmoid features
are deterministic transforms of existing clearance inputs, not new observations.
"""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
from pathlib import Path
import sys,json,time,math,hashlib,shutil
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard
D=R/'features/c6_points';X=R/'features/c10_points';O=R/'runs/c15_point_network_s20261101';O.mkdir(exist_ok=False);(O/'source').mkdir();shutil.copy2(Path(__file__),O/'source'/Path(__file__).name)
device=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8)
torch.manual_seed(20261101);np.random.seed(20261101);torch.set_float32_matmul_precision('high')
data={}
for part in ['fit','dev']:
 x=np.load(X/f'{part}_x.npy',mmap_mode='r');y=np.load(D/f'{part}_y.npy',mmap_mode='r');fs=np.load(D/f'{part}_fspl.npy',mmap_mode='r');m=np.load(D/f'{part}_mask.npy',mmap_mode='r')
 data[part]={'x':torch.from_numpy(np.array(x)).cuda(),'target':torch.from_numpy(np.array((y-fs)/20,dtype=np.float32)).cuda(),'mask':torch.from_numpy(np.array(m,dtype=np.float32)).cuda(),'fs':np.array(fs),'y':np.array(y)}
mean=data['fit']['x'].mean(0);std=data['fit']['x'].std(0).clamp_min(.01)
meta=json.loads((X/'metadata.json').read_text());ic=meta['features'].index('phase0_critical_RX');ib=meta['features'].index('phase0_margin_bin_0')
class Res(nn.Module):
 def __init__(self,w):
  super().__init__();self.fc1=nn.Linear(w,w);self.fc2=nn.Linear(w,w)
 def forward(self,x):return torch.nn.functional.silu(x+.2*self.fc2(torch.nn.functional.silu(self.fc1(x))))
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std)
  self.register_buffer('scales',torch.tensor([.01,.03,.1,.3,1.],device='cuda'));self.register_buffer('offsets',torch.tensor([-.5,0,.5],device='cuda'))
  self.first=nn.Linear(324,256);self.blocks=nn.Sequential(*[Res(256) for _ in range(3)]);self.head=nn.Linear(256,2)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):
  with torch.autocast('cuda',enabled=False):
   z=((x.float()-self.mean)/self.std).clamp(-8,8);q=torch.stack((2*torch.sinh(3*x[:,ic]),2*torch.sinh(3*x[:,ib:ib+8].amax(1))),1)
   sig=torch.sigmoid(-(q[:,:,None,None]-self.offsets[None,None,:,None])/self.scales[None,None,None,:]).flatten(1)
   h=torch.nn.functional.silu(self.first(torch.cat((z,sig),1)))
  h=self.blocks(h)
  with torch.autocast('cuda',enabled=False):
   v=self.head(h.float());return torch.stack((v[:,0]-v[:,1],v[:,0]+v[:,1]),1)
model=Model().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
config={'GPU':device,'memory_fraction_cap':.15,'seed':20261101,'inputs':'all original C10 294 features, plus30 deterministic clearance sigmoid transforms; no additional information','model':'256 width, three two-layer residual blocks, common/difference band output','batch':8192,'max_steps':8000,'evaluation_every':500,'patience_evaluations':8,'lr':.001,'final_lr':.00005,'weight_decay':1e-5,'fit_points':len(data['fit']['x']),'dev_points':len(data['dev']['x']),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'official_validation_used':False}
(O/'config.json').write_text(json.dumps(config,indent=2)+'\n')
@torch.inference_mode()
def evaluate(part,indices=None,save=False):
 model.eval();a=data[part];ids=torch.arange(len(a['x']),device='cuda') if indices is None else indices
 ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss);output=[]
 for lo in range(0,len(ids),65536):
  j=ids[lo:lo+65536]
  with torch.autocast('cuda',dtype=torch.bfloat16):pr=model(a['x'][j])
  error=(pr-a['target'][j]).double()*20;m=a['mask'][j];ss+=(error.square()*m).sum(0);cc+=m.sum(0).double()
  if save:output.append(pr.float().cpu().numpy()*20+a['fs'][j.cpu().numpy()])
 model.train()
 return (ss/cc).sqrt().cpu().tolist(),cc.cpu().tolist(),np.concatenate(output) if save else None
train_probe=torch.linspace(0,len(data['fit']['x'])-1,131072,device='cuda').long()
best=float('inf');best_step=0;history=[];start=time.time();times=[]
for step in range(1,8001):
 tic=time.perf_counter();model.train()
 j=torch.randint(len(data['fit']['x']),(8192,),device='cuda');a=data['fit'];opt.zero_grad(set_to_none=True)
 lr=.00005+.5*(.001-.00005)*(1+math.cos(math.pi*(step-1)/8000))
 for g in opt.param_groups:g['lr']=lr
 with torch.autocast('cuda',dtype=torch.bfloat16):pr=model(a['x'][j])
 m=a['mask'][j];loss=(((pr-a['target'][j]).square()*m).sum(0)/m.sum(0).clamp_min(1)).mean()
 loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
 if step<=32:
  torch.cuda.synchronize();times.append(time.perf_counter()-tic)
 if step==32:print(json.dumps({'benchmark_step_ms':1000*np.mean(times[8:]),'samples_per_second':8192/np.mean(times[8:]),'peak_GPU_GB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
 if step%500==0:
  dev,counts,_=evaluate('dev');fit,_,_=evaluate('fit',train_probe);metric=sum(v*v for v in dev)/2
  rec={'step':step,'dev_RMSE':dev,'fit_probe_RMSE':fit,'lr':lr,'seconds':time.time()-start};history.append(rec);(O/'progress.json').write_text(json.dumps(rec,indent=2)+'\n');(O/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(rec),flush=True)
  if metric<best:
   best=metric;best_step=step;torch.save({'model':model.state_dict(),'step':step,'config':config},O/'best.pt')
  if step-best_step>=4000:break
checkpoint=torch.load(O/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(checkpoint['model']);dev,counts,pred=evaluate('dev',save=True);fit,fitcounts,_=evaluate('fit')
np.save(O/'dev_prediction.npy',pred)
independent=[]
for k,b in enumerate(['415','58']):
 m=data['dev']['mask'][:,k].cpu().numpy().astype(bool);e=pred[m,k].astype(float)-data['dev']['y'][m,k].astype(float);independent.append(float(np.sqrt(np.mean(e*e))))
assert np.max(np.abs(np.array(dev)-np.array(independent)))<1e-5
ref=json.loads((R/'runs/c10_points_phase_geometry_s20261101/result.json').read_text())['bands']
gains={b:100*(1-independent[k]/ref[b]['rmse_db']) for k,b in enumerate(['415','58'])}
result={'state':'completed','best_step':best_step,'last_step':step,'DEV_RMSE':independent,'FIT_RMSE':fit,'valid_DEV_pixels':counts,'valid_FIT_pixels':fitcounts,'gain_vs_C10_percent':gains,'promote':all(g>=5 for g in gains.values()),'seconds':time.time()-start,'peak_GPU_GB':torch.cuda.max_memory_allocated()/1e9,'scope':'same sampled FIT/DEV as C10; no full-map official score','weights_sha256':hashlib.sha256((O/'best.pt').read_bytes()).hexdigest()}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
