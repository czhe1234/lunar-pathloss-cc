"""C63 controlled whole-FIT fixed vs dynamically sampled pixels."""
import sys,time,json,math,csv,hashlib,shutil,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
from src.c47_train import data,device_array,load_teacher
from src.c63_point_features import FeatureBank
from src.c61_inference import InferenceModel as Model
p=argparse.ArgumentParser();p.add_argument('--coverage',choices=['fixed','dynamic'],required=True);p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=6000);a=p.parse_args()
gpu=gpu_guard();check_source_unchanged();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.88);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
if torch.cuda.mem_get_info()[0]<64*2**30:raise RuntimeError('Need64GiB free on this approved GPU; never switch to an unapproved card.')
t0=time.time();bank=FeatureBank();print(json.dumps({'event':'bank','seconds':time.time()-t0,'GPU_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
ys=[device_array(R.parent/'data'/f'train_pl{b}.npy') for b in ['415','58']]
packed=[device_array(R.parent/'data'/f'train_mask{b}.npy') for b in ['415','58']]
fitmaps=device_array(R/'features/c38_dense/fit_map_indices.npy',dtype=torch.long)
fixed=device_array(R/'features/c38_dense/fit_pixel.npy',dtype=torch.long).reshape(-1,1024)
dev=data('dev');dev['pixels']=device_array(R/'features/c6_points/dev_pixel.npy',dtype=torch.long).reshape(-1,256);dev['maps']=device_array(R/'features/c6_points/dev_map_indices.npy',dtype=torch.long)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));fitset={rows[i]['terrain_id'] for i in fitmaps.cpu().tolist()};devset=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
assert len(fitset)==5508 and not fitset&devset
def targets(maps,pixels):
 ii=maps[:,None];y=torch.stack([z[ii,pixels//256,pixels%256].float() for z in ys],-1)
 m=torch.stack([((z[ii,pixels//8]>>(7-pixels%8))&1).float() for z in packed],-1)
 return y,m
s,c=load_teacher();torch.manual_seed(20261210);net=Model(s,'path').cuda().to(memory_format=torch.channels_last)
parent=R/'runs/c62_dense_4maps_continue3000/best.pt';ck=torch.load(parent,map_location='cpu',weights_only=False);net.load_state_dict(ck['model'])
with torch.no_grad():
 for h in net.heads:h[-1].weight.zero_();h[-1].bias.zero_()
run=R/'runs'/a.name;run.mkdir(exist_ok=False)
sources=['c63_full_train.py','c63_point_features.py','c61_inference.py','c61_model.py','c46_model.py','c52_model.py','c34_model.py']
for f in sources:shutil.copy2(R/'src'/f,run/f)
h=hashlib.sha256()
for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
cfg={**vars(a),'variant':'path','gpu':gpu,'seed':20261210,'fit_terrains':5508,'fit_maps':len(fitmaps),'batch_maps':16,'points_per_map':256,'parameters':sum(p.numel() for p in net.parameters()),'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),'initial_state_sha256':h.hexdigest(),'pretrain':str(parent.relative_to(R)),'reset':'Both final correction layers zeroed to preserve C38 initial prediction','input_features':'H/TX-derived294 features computed at requested pixels + full13channel terrain; no target/mask inputs','official_VAL_used':False,'selection':'minimum mean band RMSE on same sampledDEV612; minimum4000updates and patience2000','lr':.0005,'weight_decay':1e-4,'source_hashes':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in sources}}
atomic_json(run/'config.json',cfg)
@torch.inference_mode()
def evaluate_dev(save=None):
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss);pred=[]
 for lo in range(0,len(dev['maps']),16):
  maps=dev['maps'][lo:lo+16];B=len(maps);im=bank.image(maps);x=dev['x'][lo*256:(lo+B)*256].reshape(B,256,294);pp=dev['pixels'][lo:lo+B]
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(im,x,pp)
  j=slice(lo*256,(lo+B)*256);out=(q.reshape(-1,2).float()+dev['base'][j])*20+dev['fspl'][j];e=out.double()-dev['y'][j].double();m=dev['mask'][j]
  assert torch.isfinite(out).all();ss+=(e.square()*m).sum(0);cc+=m.sum(0).double()
  if save is not None:pred.append(out.cpu().numpy())
 net.train()
 if save is not None:np.save(save,np.concatenate(pred))
 return {'RMSE':(ss/cc).sqrt().cpu().tolist(),'SSE':ss.cpu().tolist(),'count':cc.cpu().tolist()}
monitor_pos=torch.from_numpy(np.linspace(0,len(fitmaps)-1,128,dtype=np.int64)).cuda();monitor_pixels=torch.from_numpy(np.random.default_rng(20261210).integers(0,65536,(128,256),dtype=np.int64)).cuda()
@torch.inference_mode()
def evaluate_fit():
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss)
 for lo in range(0,128,16):
  maps=fitmaps[monitor_pos[lo:lo+16]];pp=monitor_pixels[lo:lo+16];im=bank.image(maps);x,fs=bank.points(maps,pp,im);y,m=targets(maps,pp)
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(im,x,pp)
  out=(q.float()+net.last_base.reshape(16,256,2))*20+fs;e=out.double()-y.double();ss+=(e.square()*m).sum((0,1));cc+=m.sum((0,1)).double()
 net.train();return {'RMSE':(ss/cc).sqrt().cpu().tolist(),'SSE':ss.cpu().tolist(),'count':cc.cpu().tolist(),'scope':'Uniform random FIT pixels; not a heldout set'}
val=evaluate_dev();fp=evaluate_fit();assert max(abs(v-r) for v,r in zip(val['RMSE'],[2.6446351803304657,2.9987403398240975]))<1e-5
initial=val;best=float(np.mean(val['RMSE']));beststep=0;hist=[{'step':0,'DEV':val,'FIT_probe':fp,'best_step':0,'seconds':0}]
torch.save({'model':net.state_dict(),'step':0,'config':cfg},run/'best.pt');atomic_json(run/'history.json',hist)
opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0005,weight_decay=1e-4);gen=torch.Generator(device='cuda').manual_seed(20261210);t0=time.time();times=[]
for step in range(1,a.steps+1):
 ts=time.perf_counter();mi=torch.randint(len(fitmaps),(16,),device='cuda',generator=gen)
 if a.coverage=='fixed':
  pi=torch.randint(1024,(16,256),device='cuda',generator=gen);pp=fixed[mi[:,None],pi]
 else:pp=torch.randint(65536,(16,256),device='cuda',generator=gen)
 maps=fitmaps[mi];im=bank.image(maps);x,fs=bank.points(maps,pp,im);y,m=targets(maps,pp)
 lr=.0005*min(1,step/200)*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
 for group in opt.param_groups:group['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):q=net(im,x,pp)
 target=(y-fs)/20-net.last_base.reshape(16,256,2);loss=(((q.float()-target).square()*m).sum((0,1))/m.sum((0,1)).clamp_min(1)).mean()
 assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(gn);opt.step()
 if step<=30:
  torch.cuda.synchronize();times.append(time.perf_counter()-ts)
 if step==30:print(json.dumps({'event':'throughput','step_seconds':float(np.median(times[10:])),'estimated_training_minutes':float(np.median(times[10:])*a.steps/60),'GPU_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
 if step%500==0 or step==a.steps:
  val=evaluate_dev();fp=evaluate_fit()
  if np.mean(val['RMSE'])<best:
   best=float(np.mean(val['RMSE']));beststep=step;torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'best.pt')
  rec={'step':step,'DEV':val,'FIT_probe':fp,'best_step':beststep,'seconds':time.time()-t0,'lr':lr};hist.append(rec);atomic_json(run/'history.json',hist);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
  if step>=4000 and step-beststep>=2000:break
torch.save({'model':net.state_dict(),'step':step,'config':cfg},run/'last.pt');net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model'])
val=evaluate_dev(save=run/'dev_prediction.npy');fp=evaluate_fit();gain=np.array(initial['RMSE'])-val['RMSE']
out={'status':'complete','coverage':a.coverage,'scope':'all5508FITterrains; terrain-disjoint internal sampledDEV612','DEV':val,'FIT_probe':fp,'initial_C38':initial,'gain_vs_C38_db':gain.tolist(),'both_gain015db':bool((gain>=.15).all()),'best_step':beststep,'last_step':step,'seconds':time.time()-t0,'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'official_VAL_used':False}
atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)
