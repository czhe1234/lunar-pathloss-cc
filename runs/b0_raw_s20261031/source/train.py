"""Independent single-GPU training with fixed terrain split and pooled masked evaluation."""
import sys,os,fcntl,time,json,math,csv,random,argparse,shutil,signal,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import *
from tqdm import tqdm
p=argparse.ArgumentParser()
p.add_argument('--run',required=True);p.add_argument('--mode',choices=['raw','geometry','profile'],default='raw');p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=.001);p.add_argument('--width',type=int,default=32);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--part',choices=['fit','full'],default='fit');p.add_argument('--patience',type=int,default=10);p.add_argument('--compile',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--init-checkpoint');args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=ROOT/'runs'/args.run
if out.exists() and not args.resume:raise RuntimeError(f'Run already exists: {out}')
out.mkdir(parents=True,exist_ok=True)
stop_requested=False

def request_stop(signum,frame):
 global stop_requested;stop_requested=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
def checkpoint(path,obj):
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
try:
 info=gpu_guard();source=check_source_unchanged()
 torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 config=vars(args)|{'device':info,'output_db_offset':OFFSET,'output_db_scale':SCALE,'precision':'BF16 convolutions with FP32 head/loss; FP64 score accumulation','loss':'equal mean of separately normalized masked MSE per band','selection_metric':'equal mean dB MSE on development set; not official normalized score','data_fingerprints':source['files'],'internal_split':json.loads((ROOT/'audit/internal_split.json').read_text()),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'src/cc_core.py',Path(__file__),ROOT/'run_gpu3.sh']}}
 if not args.resume:
  atomic_json(out/'config.json',config)
  (out/'source').mkdir(exist_ok=True)
  for f in [ROOT/'src/cc_core.py',Path(__file__),ROOT/'run_gpu3.sh']:shutil.copy2(f,out/'source'/f.name)
 data=ResidentData('train');train_idx=data.indices(args.part)
 evaluation_data=data if args.part=='fit' else ResidentData('val')
 val_idx=data.indices('dev') if args.part=='fit' else evaluation_data.indices('full')
 if args.mode=='profile':
  data.profile=torch.load(ROOT/'features/train_profile.pt',map_location='cuda',weights_only=True)
  if args.part=='full':evaluation_data.profile=torch.load(ROOT/'features/val_profile.pt',map_location='cuda',weights_only=True)
 model=model_for(args.mode,args.width)
 if args.init_checkpoint:model.load_state_dict(torch.load(args.init_checkpoint,map_location='cuda',weights_only=False)['model'])
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001,fused=True)
 start_epoch=1;best=float('inf');best_epoch=0;history=[]
 if args.resume:
  ck=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start_epoch=ck['epoch']+1;best=ck['best'];best_epoch=ck['best_epoch'];history=json.loads((out/'history.json').read_text())
 net=torch.compile(model) if args.compile else model
 nparams=sum(t.numel() for t in model.parameters());tstart=time.time()
 print(json.dumps({'event':'training_start','run':args.run,'train_maps':len(train_idx),'eval_maps':len(val_idx),'parameters':nparams,'batch':args.batch,'mode':args.mode,'device':info}),flush=True)
 for epoch in range(start_epoch,args.epochs+1):
  epstart=time.time();rng=np.random.default_rng(args.seed+epoch);order=rng.permutation(train_idx);num_steps=math.ceil(len(order)/args.batch)
  lr=args.lr*min(1.,epoch/3.)*(.01+.99*(1+math.cos(math.pi*max(0,epoch-3)/max(1,args.epochs-3)))/2)
  for group in opt.param_groups:group['lr']=lr
  model.train();train_sum=torch.zeros((),device='cuda');seen=0
  bar=tqdm(range(num_steps),desc=f'{args.run} epoch {epoch}/{args.epochs}',mininterval=2.,dynamic_ncols=False,ncols=100)
  for step in bar:
   ii=order[step*args.batch:(step+1)*args.batch]
   x,y,m=data.batch(ii,args.mode,int(rng.integers(0,8)));opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x);loss=masked_loss(pred,y,m)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,foreach=True,error_if_nonfinite=True);opt.step()
   train_sum+=loss.detach()*len(ii);seen+=len(ii)
   if step%20==0 or step+1==num_steps:
    avg=float(train_sum)/seen*SCALE*SCALE
    bar.set_postfix(mse_db=f'{avg:.2f}',refresh=False)
    atomic_json(out/'progress.json',{'status':'training','epoch':epoch,'epochs':args.epochs,'completed_batches':step+1,'total_batches':num_steps,'completed_maps':seen,'train_mse_db_batch_weighted':avg,'elapsed_seconds':time.time()-tstart,'gpu_uuid':GPU_UUID,'pid':os.getpid(),'updated_unix':time.time()})
   if stop_requested or (out/'STOP').exists():break
  torch.cuda.synchronize();train_seconds=time.time()-epstart
  if stop_requested or (out/'STOP').exists():
   checkpoint(out/'interrupted.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'completed_batches':step+1,'args':vars(args)})
   atomic_json(out/'progress.json',{'status':'stopped','epoch':epoch,'completed_batches':step+1,'pid':os.getpid()});break
  atomic_json(out/'progress.json',{'status':'evaluating','epoch':epoch,'epochs':args.epochs,'eval_maps':len(val_idx),'pid':os.getpid(),'updated_unix':time.time()})
  metrics=evaluate(model,evaluation_data,val_idx,args.mode,args.batch)
  record={'epoch':epoch,'lr':lr,'train_mse_db_batch_weighted':float(train_sum)/seen*SCALE*SCALE,'train_seconds':train_seconds,'training_maps_per_second':seen/train_seconds,'epoch_seconds':time.time()-epstart,**metrics}
  history.append(record);atomic_json(out/'history.json',history)
  if metrics['mse_mean']<best:
   best=metrics['mse_mean'];best_epoch=epoch
   checkpoint(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'metrics':metrics,'args':vars(args),'source_sha256':config['source_sha256']})
  checkpoint(out/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'args':vars(args)})
  if epoch%10==0:checkpoint(out/f'epoch_{epoch:03d}.pt',{'model':model.state_dict(),'epoch':epoch,'metrics':metrics,'args':vars(args)})
  print('EPOCH_RESULT '+json.dumps(record),flush=True)
  atomic_json(out/'progress.json',{'status':'epoch_complete','epoch':epoch,'epochs':args.epochs,'best_epoch':best_epoch,'best_mse_mean':best,'last_rmse415':metrics['rmse415'],'last_rmse58':metrics['rmse58'],'elapsed_seconds':time.time()-tstart,'pid':os.getpid(),'updated_unix':time.time()})
  if epoch-best_epoch>=args.patience:
   print(f'Early stopping after epoch {epoch}; best epoch {best_epoch}',flush=True);break
 else:pass
 if not stop_requested and not (out/'STOP').exists():
  ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
  final=evaluate(model,evaluation_data,val_idx,args.mode,args.batch,save_per_map=out/'best_per_map.csv')
  final.update({'run':args.run,'best_epoch':best_epoch,'completed_epochs':len(history),'evaluation_split':'development' if args.part=='fit' else 'official_validation','goal_met_on_official_validation':args.part=='full' and final['rmse415']<=1 and final['rmse58']<=1,'total_seconds':time.time()-tstart,'parameters':nparams,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9})
  atomic_json(out/'result.json',final);atomic_json(out/'progress.json',{'status':'completed',**final,'pid':os.getpid(),'updated_unix':time.time()})
  print('FINAL_RESULT '+json.dumps(final),flush=True)
except Exception as exc:
 atomic_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'time':time.time(),'pid':os.getpid()})
 raise
