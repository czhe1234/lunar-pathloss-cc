"""Function-preserving width expansion vs matched C38 continuation."""
import sys,time,json,math,argparse,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
from src.c34_model import Model
from src.c47_train import device_array
p=argparse.ArgumentParser();p.add_argument('--factor',type=int,choices=[1,2],required=True);p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=20000);p.add_argument('--lr',type=float,default=.0001);a=p.parse_args()
gpu=gpu_guard();check_source_unchanged();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.78)
torch.manual_seed(20261207);np.random.seed(20261207);torch.set_float32_matmul_precision('high')
ck=torch.load(R/'runs/c38_1024_continue/best.pt',map_location='cpu',weights_only=False);s=ck['model'];c=ck['config']
def make(factor):
 net=Model(c['mode'],c['members'],c['width']*factor,s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda()
 out={k:v.clone() for k,v in s.items()}
 if factor>1:
  for j in range(3):
   q='layers.'+str(j)+'.'
   out[q+'weight']=s[q+'weight'].repeat(factor,1 if j==0 else factor)/(1 if j==0 else factor)
   out[q+'r']=s[q+'r'].repeat(1,1 if j==0 else factor)
   out[q+'s']=s[q+'s'].repeat(1,factor);out[q+'bias']=s[q+'bias'].repeat(1,factor)
  out['head']=s['head'].repeat(1,factor,1)/factor
 net.load_state_dict(out);return net
data={}
for part,folder,label in [('fit','c38_dense','c38_dense'),('dev','c10_points','c6_points')]:
 d={'x':device_array(R/f'features/{folder}/{part}_x.npy',dtype=torch.float32)}
 for k in ['y','fspl','mask']:d[k]=device_array(R/f'features/{label}/{part}_{k}.npy',dtype=torch.float32)
 d['target']=(d['y']-d['fspl'])/20;data[part]=d
net=make(a.factor);net.eval();ref=make(1);ref.eval()
torch.set_float32_matmul_precision('highest')
with torch.inference_mode():
 delta=(net(data['dev']['x'][:256])-ref(data['dev']['x'][:256])).abs().max().item()*20
assert delta<.001,delta
del ref;torch.cuda.empty_cache();torch.set_float32_matmul_precision('high')
run=R/'runs'/a.name;run.mkdir(exist_ok=False);shutil.copy2(__file__,run/'train_source.py');shutil.copy2(R/'src/c34_model.py',run/'model_source.py')
cfg={**c,**vars(a),'width':c['width']*a.factor,'gpu':gpu,'seed':20261207,'function_preserving_initial_max_diff_db':delta,'parameters':sum(p.numel() for p in net.parameters()),'loss':'mean individual-member masked two-band MSE, as C38','official_VAL_used':False,'parent_checkpoint_sha256':hashlib.sha256((R/'runs/c38_1024_continue/best.pt').read_bytes()).hexdigest(),'selection':'minimum mean band RMSE on unchanged sampled internal DEV612; patience5000'}
atomic_json(run/'config.json',cfg)
probe=torch.linspace(0,len(data['fit']['x'])-1,131072,device='cuda').long()
@torch.inference_mode()
def evaluate(part,ids=None,save=False):
 d=data[part];n=len(d['x']) if ids is None else len(ids);ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss);pr=[];net.eval()
 for lo in range(0,n,4096):
  j=slice(lo,min(lo+4096,n)) if ids is None else ids[lo:lo+4096]
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(d['x'][j]).mean(1).float()
  out=q*20+d['fspl'][j];e=out.double()-d['y'][j].double();m=d['mask'][j]
  ss+=(e.square()*m).sum(0);cc+=m.sum(0).double()
  if save:pr.append(out.cpu().numpy())
 net.train()
 return {'RMSE':(ss/cc).sqrt().cpu().tolist(),'SSE':ss.cpu().tolist(),'count':cc.cpu().tolist()},np.concatenate(pr) if save else None
val,_=evaluate('dev');fp,_=evaluate('fit',probe);initial=val
assert max(abs(v-r) for v,r in zip(val['RMSE'],[2.6446351803304657,2.9987403398240975]))<.005,val
best=np.mean(val['RMSE']);beststep=0
torch.save({'model':net.state_dict(),'step':0,'config':cfg},run/'best.pt')
hist=[{'step':0,'DEV':val,'FIT_probe':fp,'seconds':0.,'best_step':0}]
atomic_json(run/'history.json',hist)
opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
gen=torch.Generator(device='cuda').manual_seed(20261207);t0=time.time()
for step in range(1,a.steps+1):
 j=torch.randint(len(data['fit']['x']),(4096,),device='cuda',generator=gen);d=data['fit']
 lr=a.lr*min(1,step/200)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
 for g in opt.param_groups:g['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):q=net(d['x'][j])
 mm=d['mask'][j,None,:]
 loss=(((q.float()-d['target'][j,None,:]).square()*mm).sum((0,1))/(mm.sum((0,1))*8).clamp_min(1)).mean()
 assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
 if step%1000==0 or step==a.steps:
  val,_=evaluate('dev');fp,_=evaluate('fit',probe)
  if np.mean(val['RMSE'])<best:
   best=float(np.mean(val['RMSE']));beststep=step;torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'best.pt')
  rec={'step':step,'DEV':val,'FIT_probe':fp,'best_step':beststep,'lr':lr,'seconds':time.time()-t0};hist.append(rec);atomic_json(run/'history.json',hist);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
  if step>=6000 and step-beststep>=5000:break
torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'last.pt')
net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model']);val,pr=evaluate('dev',save=True);fp,_=evaluate('fit',probe);np.save(run/'dev_prediction.npy',pr)
out={'status':'complete','DEV':val,'FIT_probe':fp,'initial':initial,'gain_vs_C38_db':(np.array([2.6446351803304657,2.9987403398240975])-val['RMSE']).tolist(),'best_step':beststep,'last_step':step,'seconds':time.time()-t0,'parameters':cfg['parameters'],'gpu':gpu,'official_VAL_used':False}
atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)
