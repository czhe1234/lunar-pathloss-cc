"""C61 whole-FIT learned terrain fields; matched local/path readout."""
import sys,time,json,math,argparse,hashlib,shutil,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
from torch.nn import functional as F
from src.c47_train import load_teacher,data,device_array,tiny_selection
from src.c49_train import terrain_fields
from src.c61_model import Model
A=R/'audit/c61_global_terrain_20261102'
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['local','path'],required=True);p.add_argument('--scope',choices=['tiny','full'],required=True);p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch-maps',type=int,default=16);p.add_argument('--points',type=int,default=256);a=p.parse_args()
gpu=gpu_guard();check_source_unchanged();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.88);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
torch.manual_seed(20261208);np.random.seed(20261208);s,c=load_teacher()
fields,info=terrain_fields();margins=device_array(R/'features/c49_visibility/train_margins.npy')
if a.scope=='tiny':
 selection,maps,terrains=tiny_selection();fit=data('fit',selection);dev=fit;fit['pixels']=device_array(R/'features/c38_dense/fit_pixel.npy',selection,torch.long)
 fit['maps']=torch.tensor(maps,device='cuda',dtype=torch.long);fit['P']=1024;steps=a.steps;a.batch_maps=4
else:
 fit=data('fit');dev=data('dev');steps=a.steps
 for part,d in [('fit',fit),('dev',dev)]:
  lab='c38_dense' if part=='fit' else 'c6_points'
  d['pixels']=device_array(R/f'features/{lab}/{part}_pixel.npy',dtype=torch.long)
  d['maps']=device_array(R/f'features/{lab}/{part}_map_indices.npy',dtype=torch.long);d['P']=1024 if part=='fit' else 256
 maps=fit['maps'].cpu().tolist();rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));terrains=sorted({rows[i]['terrain_id'] for i in maps})
 assert len(terrains)==5508 and not set(terrains)&set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
yy,xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij')
def image(maps):
 inf=info[maps];B=len(maps);v=fields[inf[:,0]];h=v[...,0];sx=v[...,1];sy=v[...,2];rr=inf[:,1];cc=inf[:,2];ht=h[torch.arange(B,device='cuda'),rr,cc]
 dx=xx[None]-cc[:,None,None];dy=yy[None]-rr[:,None,None];dz=h+1-ht[:,None,None]-3;d=torch.sqrt(dx.square()+dy.square()+dz.square()).clamp_min(.25)
 hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0];em=margins[maps].float()
 vals=[h/100,(h-ht[:,None,None])/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(d)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx.square()+dy.square())/8),torch.asinh(em[...,0]/2)/3,torch.asinh(em[...,1]/2)/3]
 return torch.stack(vals,1).contiguous(memory_format=torch.channels_last)
warm=torch.load(R/'runs/c52_adapter_full/best.pt',map_location='cpu',weights_only=False)['model']
torch.manual_seed(20261208);net=Model(s,a.variant,warm).cuda().to(memory_format=torch.channels_last)
run=R/'runs'/a.name;run.mkdir(exist_ok=False)
for f in ['c61_train.py','c61_model.py','c46_model.py','c34_model.py','c52_model.py']:shutil.copy2(R/'src'/f,run/f)
cfg={**vars(a),'gpu':gpu,'seed':20261208,'fit_terrains':len(terrains),'fit_maps':len(maps),'steps':steps,'fit_rows':len(fit['x']),'parameters':sum(p.numel() for p in net.parameters()),'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),'official_VAL_used':False,'inputs':'H/TX-derived13channel full image + existing294features + frozenC38latent','loss':'C38 residual masked two-band MSE','initialization':'C52 global projection only; zero new output heads; identical local/path parameter initializations','source_hashes':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['c61_model.py','c61_train.py','c46_model.py','c34_model.py','c52_model.py']}}
h=hashlib.sha256()
for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
cfg['initial_state_sha256']=h.hexdigest();atomic_json(run/'config.json',cfg)
@torch.inference_mode()
def evaluate(d,positions=None,save=None):
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cnt=torch.zeros_like(ss);pred=[]
 ids=torch.arange(len(d['maps']),device='cuda') if positions is None else positions
 for lo in range(0,len(ids),a.batch_maps):
  ids0=ids[lo:lo+a.batch_maps];B=len(ids0);maps0=d['maps'][ids0];im=image(maps0)
  with torch.autocast('cuda',dtype=torch.bfloat16):st=net.encode(im)
  for start in range(0,d['P'],256):
   pp=torch.arange(start,min(d['P'],start+256),device='cuda');j=ids0[:,None]*d['P']+pp[None];pix=d['pixels'][j]
   with torch.autocast('cuda',dtype=torch.bfloat16):q=net.decode(st,d['x'][j],pix)
   out=(q.float()+d['base'][j])*20+d['fspl'][j];e=out.double()-d['y'][j].double();m=d['mask'][j]
   assert torch.isfinite(out).all();ss+=(e.square()*m).sum((0,1));cnt+=m.sum((0,1)).double()
   if save is not None:pred.append((j.flatten().cpu().numpy(),out.reshape(-1,2).cpu().numpy()))
 net.train();res={'RMSE':(ss/cnt).sqrt().cpu().tolist(),'SSE':ss.cpu().tolist(),'count':cnt.cpu().tolist()}
 if save is not None:
  out=np.empty((len(d['x']),2),np.float32)
  for ix,pr in pred:out[ix]=pr
  np.save(save,out)
 return res
fitprobe=torch.from_numpy(np.linspace(0,len(fit['maps'])-1,min(128,len(fit['maps'])),dtype=np.int64)).cuda()
val=evaluate(dev);fp=evaluate(fit,fitprobe)
if a.scope=='full':assert max(abs(v-r) for v,r in zip(val['RMSE'],[2.6446351803304657,2.9987403398240975]))<1e-5,val
best=float(np.mean(val['RMSE']));beststep=0;initial=val;hist=[{'step':0,'DEV':val,'FIT_probe':fp,'seconds':0,'best_step':0}]
torch.save({'model':net.state_dict(),'step':0,'config':cfg},run/'best.pt');atomic_json(run/'history.json',hist)
opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.001,weight_decay=1e-4);gen=torch.Generator(device='cuda').manual_seed(20261208);t0=time.time();times=[]
interval=200 if a.scope=='tiny' else 500
for step in range(1,steps+1):
 t=time.perf_counter();mi=torch.randint(len(fit['maps']),(a.batch_maps,),device='cuda',generator=gen);pi=torch.randint(fit['P'],(a.batch_maps,a.points),device='cuda',generator=gen);j=mi[:,None]*fit['P']+pi
 im=image(fit['maps'][mi]);pix=fit['pixels'][j];lr=.001*min(1,step/(50 if a.scope=='tiny' else 200))*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/steps)))
 for g in opt.param_groups:g['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):q=net(im,fit['x'][j],pix)
 y=fit['target'][j]-fit['base'][j];m=fit['mask'][j];loss=(((q.float()-y).square()*m).sum((0,1))/m.sum((0,1)).clamp_min(1)).mean()
 assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(gn);opt.step()
 if step<=20:
  torch.cuda.synchronize();times.append(time.perf_counter()-t)
 if step==20:
  grads={k:(float(p.grad.norm()) if p.grad is not None else None) for k,p in net.named_parameters() if k in ['encoder.stem.0.weight','offset.weight','kv.0.weight','global_in.weight']}
  assert grads['encoder.stem.0.weight']>0 and grads['global_in.weight']>0
  if a.variant=='path':assert grads['offset.weight']>0 and grads['kv.0.weight']>0
  print(json.dumps({'event':'throughput','step_seconds':float(np.median(times[10:])),'estimated_minutes':float(np.median(times[10:])*steps/60),'gradients':grads,'gpu_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
 if step%interval==0 or step==steps:
  val=evaluate(dev);fp=evaluate(fit,fitprobe)
  if np.mean(val['RMSE'])<best:
   best=float(np.mean(val['RMSE']));beststep=step;torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'best.pt')
  rec={'step':step,'DEV':val,'FIT_probe':fp,'seconds':time.time()-t0,'lr':lr,'best_step':beststep};hist.append(rec);atomic_json(run/'history.json',hist);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
  if a.scope=='full' and step>=3000 and step-beststep>=2500:break
torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'last.pt');net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model']);val=evaluate(dev,save=run/'dev_prediction.npy');fp=evaluate(fit,fitprobe)
out={'status':'complete','scope':a.scope,'DEV':val,'FIT_probe':fp,'initial':initial,'gain_vs_C38_db':(np.array(initial['RMSE'])-val['RMSE']).tolist(),'best_step':beststep,'last_step':step,'seconds':time.time()-t0,'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'tiny_both_le_080db':max(fp['RMSE'])<=.8,'both_sampledDEV_gain015db':bool(a.scope=='full' and min(np.array(initial['RMSE'])-val['RMSE'])>=.15),'official_VAL_used':False,'evaluation_scope':'FIT only; DEV is aliasFIT' if a.scope=='tiny' else 'terrain-disjoint sampled internal DEV612'}
atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)
