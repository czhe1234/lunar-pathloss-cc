"""Full-FIT cooperative independent-expert optimization; C38 remains fallback."""
import sys,time,math,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
from src.c47_train import data,load_teacher,teacher_net
from src.c50_model import Model
from tqdm import trange
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.78);torch.set_float32_matmul_precision('high')
torch.manual_seed(20261102);check_source_unchanged()
run=R/'runs/c50_cooperative';run.mkdir(exist_ok=False)
for name in ['c50_train.py','c50_model.py','c47_train.py','c34_model.py']:shutil.copy2(R/'src'/name,run/name)
state,cfg=load_teacher();fit=data('fit');dev=data('dev')
teacher=teacher_net(state,cfg);net=Model(state).cuda().train()
# Verify algebraic conversion independently in ordinary float32.
x=dev['x'][:128];net.eval();torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode():
 ref=teacher(x).mean(1);new=net(x);identity_max=float((ref-new).abs().max()*20)
assert identity_max<.001,identity_max
torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):bf16_rms=((teacher(x).mean(1)-net(x)).square().mean(0).sqrt()*20).tolist()
del teacher;torch.cuda.empty_cache()
config={'gpu':gpu,'seed':20261102,'parameters':sum(p.numel() for p in net.parameters()),'batch':4096,'steps':12000,
 'input_features':294,'independent_experts':8,'width':1024,'hidden_layers':3,'dropout':.02,'initial_lr':{'expert':.0003,'embedding':.0001,'gate':.001},
 'parent':'runs/c38_1024_continue/best.pt','initial_FP32_identity_max_db':identity_max,'initial_BF16_branch_layout_RMS_difference_db':bf16_rms,
 'fit_rows':len(fit['x']),'dev_rows':len(dev['x']),'official_VAL_used':False,
 'selection':'Minimum mean two-band sampled DEV RMSE; original C38 remains global fallback if new model is worse',
 'source_hashes':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['c50_train.py','c50_model.py','c47_train.py','c34_model.py']}}
atomic_json(run/'config.json',config)
@torch.inference_mode()
def evaluate(d,ids=None,save=None):
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss);pred=[]
 n=len(d['x']) if ids is None else len(ids)
 for lo in range(0,n,4096):
  j=slice(lo,min(lo+4096,n)) if ids is None else ids[lo:lo+4096]
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(d['x'][j])
  pr=q*20+d['fspl'][j];e=pr.double()-d['y'][j].double();m=d['mask'][j]
  assert torch.isfinite(pr).all();ss+=(e.square()*m).sum(0);cc+=m.sum(0).double()
  if save is not None:pred.append(pr.cpu().numpy())
 rm=(ss/cc).sqrt().cpu().tolist();net.train()
 if save is not None:np.save(save,np.concatenate(pred))
 return {'RMSE':rm,'combined':sum(rm)/2,'SSE':ss.cpu().tolist(),'count':cc.cpu().tolist()}
probe=torch.from_numpy(np.linspace(0,len(fit['x'])-1,65536,dtype=np.int64)).cuda()
initial=evaluate(dev);hist=[{'step':0,'DEV':initial,'FIT':evaluate(fit,probe)}];best=initial['combined'];beststep=0
atomic_json(run/'history.json',hist);torch.save({'model':net.state_dict(),'step':0,'config':config},run/'best.pt')
groups=[{'params':[p for n,p in net.named_parameters() if not n.startswith(('embedding.','gate.'))],'lr':.0003,'base_lr':.0003},
 {'params':net.embedding.parameters(),'lr':.0001,'base_lr':.0001},{'params':net.gate.parameters(),'lr':.001,'base_lr':.001}]
opt=torch.optim.AdamW(groups,weight_decay=1e-4)
gen=torch.Generator(device='cuda').manual_seed(20261102);t0=time.time();times=[]
for step in trange(1,12001,desc='C50 cooperative',mininterval=10):
 ts=time.perf_counter();j=torch.randint(len(fit['x']),(4096,),device='cuda',generator=gen)
 factor=min(1,step/200)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/12000)))
 for g in opt.param_groups:g['lr']=g['base_lr']*factor
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):pr=net(fit['x'][j])
 mm=fit['mask'][j];loss=(((pr-fit['target'][j]).square()*mm).sum(0)/mm.sum(0).clamp_min(1)).mean()
 assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),10);assert torch.isfinite(gn);opt.step()
 if step<=30:torch.cuda.synchronize();times.append(time.perf_counter()-ts)
 if step==30:print(json.dumps({'event':'throughput','step_seconds':float(np.median(times[10:])),'estimated_training_minutes':float(np.median(times[10:])*12000/60)}),flush=True)
 if step%1000==0:
  val=evaluate(dev);fp=evaluate(fit,probe)
  if val['combined']<best:
   best=val['combined'];beststep=step;torch.save({'model':net.state_dict(),'step':step,'config':config},run/'best.pt')
  rec={'step':step,'steps':12000,'DEV':val,'FIT':fp,'best_step':beststep,'seconds':time.time()-t0};hist.append(rec)
  atomic_json(run/'history.json',hist);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
  if step>=4000 and step-beststep>=3000:break
torch.save({'model':net.state_dict(),'step':step,'config':config},run/'last.pt')
net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model'])
val=evaluate(dev,save=run/'dev_prediction.npy');fp=evaluate(fit,probe)
base=np.array([2.6446351803304657,2.9987403398240975]);gain=base-np.array(val['RMSE'])
result={'status':'complete','initial':initial,'DEV':val,'FIT_probe':fp,'gain_vs_originalC38_db':gain.tolist(),'both_gain010db':bool((gain>=.10).all()),
 'best_step':beststep,'last_step':step,'seconds':time.time()-t0,'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'official_VAL_used':False}
atomic_json(run/'result.json',result);print(json.dumps(result),flush=True)
