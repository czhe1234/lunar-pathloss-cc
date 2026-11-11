import sys,os,json,csv,time,argparse,fcntl,hashlib,shutil,signal,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,atomic_json,check_source_unchanged,ResidentData
from c3_geometry import make_inputs,geometry
from c3_model import Model
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',type=int,default=32);p.add_argument('--width',type=int,default=32);p.add_argument('--lr',type=float,default=.0006);p.add_argument('--patience',type=int,default=12);p.add_argument('--compile',action='store_true');p.add_argument('--part',choices=['fit','full'],default='fit');p.add_argument('--init');p.add_argument('--resume',action='store_true');args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();check_source_unchanged();torch.set_num_threads(8);torch.manual_seed(20261031);np.random.seed(20261031)
torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
run=ROOT/'runs'/args.run;run.mkdir(exist_ok=True);source=run/'source';source.mkdir(exist_ok=True)
files=[Path(__file__),ROOT/'src/cc_core.py',ROOT/'src/c3_geometry.py',ROOT/'src/c3_model.py']
shas={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
config={**vars(args),'device':info,'source_sha256':shas,'seed':20261031,'input_channels':27,'geometry_channels':16,'D4_augmentation':False,'pair_loss_weight':.25,'pair_batches_every':4,'pair_neighbor_count':8,'output':'FSPL at 415 MHz and 5.775 GHz plus 20 times learned residual','official_validation_used_for_selection':False}
if args.resume:
 prev=json.loads((run/'config.json').read_text());assert prev['source_sha256']==shas
else:
 assert not (run/'config.json').exists(),'Existing run; use resume'
 for f in files:shutil.copy2(f,source/f.name)
 atomic_json(run/'config.json',config)
stop=False

def signal_stop(*unused):
 global stop;stop=True
signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
data=ResidentData('train');fit=data.indices(args.part);dev=data.indices('dev') if args.part=='fit' else np.array([],dtype=np.int64)
cachemeta=json.loads((ROOT/'features/c3/train.json').read_text());assert cachemeta['complete'] and cachemeta['source_sha256']==shas['c3_geometry.py']
raw=np.load(ROOT/'features/c3/train.npy',mmap_mode='r');geo=torch.empty(raw.shape,dtype=torch.float16,device='cuda')
for lo in range(0,len(raw),256):geo[lo:lo+256].copy_(torch.from_numpy(np.array(raw[lo:lo+256],copy=True)))
del raw;torch.cuda.synchronize();print(json.dumps({'event':'all_inputs_loaded','memory_allocated_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
parity=[]
for i in [0,51,500]:
 idx=torch.tensor([i],device='cuda');fresh=geometry(data.h[data.hmrow[idx]],data.txr[idx],data.txc[idx]);delta=float((fresh-geo[idx]).abs().max());assert delta<.002;parity.append({'index_row':i,'max_abs_difference':delta})
atomic_json(run/'input_cache_replay_check.json',{'passed':True,'results':parity})

model=Model(args.width).to('cuda',memory_format=torch.channels_last)
if args.init:
 ck=torch.load(ROOT/'runs'/args.init/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4,fused=True)
first=1;best=float('inf');bestepoch=0;hist=[]
if args.resume:
 ck=torch.load(run/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);first=ck['epoch']+1;best=ck['best'];bestepoch=ck['best_epoch'];hist=json.loads((run/'history.json').read_text())
net=torch.compile(model) if args.compile else model
pool={}
for i in fit:pool.setdefault(data.rows[i]['hm_row'],[]).append(int(i))
neighbors={}
for ids in pool.values():
 if len(ids)<2:continue
 xy=np.array([[int(data.rows[i]['tx_row']),int(data.rows[i]['tx_col'])] for i in ids]);dist=((xy[:,None]-xy[None])**2).sum(-1);order=np.argsort(dist,axis=1)[:,1:9]
 for j,i in enumerate(ids):neighbors[i]=np.array(ids)[order[j]]

def batch(ii):
 idx=torch.as_tensor(ii,device='cuda',dtype=torch.long)
 x,fs=make_inputs(data.h[data.hmrow[idx]],data.txr[idx],data.txc[idx],geo[idx])
 return x,fs,data.y[idx].float(),data.mask[idx].float()

@torch.inference_mode()
def evaluate(indices):
 model.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss)
 for lo in range(0,len(indices),args.batch):
  x,fs,y,m=batch(indices[lo:lo+args.batch])
  with torch.autocast('cuda',dtype=torch.bfloat16):pr=model(x)+fs
  ss+=((pr-y).double().square()*m).sum((0,2,3));cc+=m.sum((0,2,3),dtype=torch.float64)
 rm=(ss/cc).sqrt().cpu().tolist();model.train();return {'rmse415':rm[0],'rmse58':rm[1],'mse_mean':sum(v*v for v in rm)/2,'valid_pixels':cc.cpu().tolist(),'maps_per_band':len(indices)}

def save(name,epoch):
 dst=run/name;temp=dst.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'best_epoch':bestepoch,'config':config},temp);temp.replace(dst)

t0=time.time()
for epoch in range(first,args.epochs+1):
 rng=np.random.default_rng(20261031+epoch);order=rng.permutation(fit);lo=0;step=0;main_sum=0;pair_sum=0;nseen=0;te=time.time();model.train()
 while lo<len(order):
  paired=step%4==0
  take=args.batch//2 if paired else args.batch;anchor=order[lo:lo+take];lo+=len(anchor);ii=anchor;npairs=0
  if paired:
   partners=np.array([rng.choice(neighbors[int(i)]) if int(i) in neighbors else int(i) for i in anchor]);ii=np.concatenate((anchor,partners));npairs=len(anchor)
  progress=(epoch-1)+lo/len(order);warm=min(1.,progress/2);lr=args.lr*warm*(.02+.98*.5*(1+math.cos(math.pi*progress/args.epochs)))
  for g in opt.param_groups:g['lr']=lr
  x,fs,y,m=batch(ii);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pr=net(x)+fs
  error=(pr-y)/20;main=(error.square().mul(m).sum((0,2,3))/m.sum((0,2,3)).clamp_min(1)).mean();pair=main.new_zeros(())
  if npairs:
   joint=m[:npairs]*m[npairs:];delta=error[:npairs]-error[npairs:]
   pair=(delta.square().mul(joint).sum((0,2,3))/joint.sum((0,2,3)).clamp_min(1)).mean()
  loss=main+.25*pair
  if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
  main_sum+=main.detach()*len(ii);pair_sum+=pair.detach()*len(ii);nseen+=len(ii);step+=1
  if step%100==0:
   torch.cuda.synchronize();atomic_json(run/'progress.json',{'state':'training','epoch':epoch,'anchor_maps_done':lo,'anchor_maps_total':len(order),'processed_maps':nseen,'seconds_in_epoch':time.time()-te,'main_rmse_proxy':float((main_sum/nseen).sqrt()*20),'best_epoch':bestepoch,'best_mse':best if math.isfinite(best) else None,'memory_peak_GB':torch.cuda.max_memory_allocated()/1e9})
  if stop or (run/'STOP').exists():
   save('interrupted.pt',epoch);atomic_json(run/'progress.json',{'state':'interrupted','epoch':epoch,'anchor_maps_done':lo});sys.exit(2)
 torch.cuda.synchronize();trainsec=time.time()-te
 result=evaluate(dev) if len(dev) else None
 row={'epoch':epoch,'lr':lr,'train_rmse_proxy':float((main_sum/nseen).sqrt()*20),'pair_mse_normalized':float(pair_sum/nseen),'processed_maps':nseen,'train_seconds':trainsec,'elapsed_seconds':time.time()-t0,'dev':result,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9}
 hist.append(row);atomic_json(run/'history.json',hist)
 score=result['mse_mean'] if result else 0
 if score<best or not len(dev):best=score;bestepoch=epoch;save('best.pt',epoch)
 save('last.pt',epoch);atomic_json(run/'progress.json',{'state':'training','latest':row,'best_epoch':bestepoch,'best_mse':best});print(json.dumps(row),flush=True)
 if len(dev) and epoch-bestepoch>=args.patience:break
ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
final=evaluate(dev) if len(dev) else {'maps_per_band':len(fit)}
final.update({'state':'completed','run':args.run,'best_epoch':bestepoch,'epochs_completed':hist[-1]['epoch'],'evaluation_split':'development' if len(dev) else 'full_training_checkpoint_not_scored','part':args.part,'seconds':time.time()-t0})
atomic_json(run/'result.json',final);atomic_json(run/'progress.json',final);print(json.dumps(final),flush=True)
