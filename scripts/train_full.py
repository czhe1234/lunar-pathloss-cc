"""Full-training refit with a dev-selected fixed stopping epoch. Never reads val targets."""
import sys,os,fcntl,time,json,math,random,argparse,shutil,signal,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import *
from tqdm import tqdm
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--selected-dev-run',required=True);p.add_argument('--seed',type=int,default=20261031);p.add_argument('--resume',action='store_true');args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=ROOT/'runs'/args.run
if out.exists() and not args.resume:raise RuntimeError(f'Run exists: {out}')
out.mkdir(parents=True,exist_ok=True);stop_requested=False

def request_stop(signum,frame):
 global stop_requested;stop_requested=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
def checkpoint(path,obj):
 tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
try:
 dev_run=ROOT/'runs'/args.selected_dev_run;selected=json.loads((dev_run/'config.json').read_text());dev_result=json.loads((dev_run/'result.json').read_text())
 assert selected['part']=='fit' and dev_result['evaluation_split']=='development'
 independent=json.loads((dev_run/'predictions/train_dev_bf16/independent_score.json').read_text());assert independent['all_verification_checks_passed']
 mode=selected['mode'];width=selected['width'];batch=selected['batch'];schedule_epochs=selected['epochs'];stop_epoch=dev_result['best_epoch'];base_lr=selected['lr']
 info=gpu_guard();source=check_source_unchanged();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 files=[ROOT/'src/cc_core.py',Path(__file__),ROOT/'run_gpu3.sh']
 config={'run':args.run,'mode':mode,'width':width,'batch':batch,'epochs':schedule_epochs,'fixed_stop_epoch':stop_epoch,'lr':base_lr,'seed':args.seed,'part':'full','compile':selected['compile'],'selected_dev_run':args.selected_dev_run,'selected_dev_result':dev_result,'selection_rule':'Architecture, hyperparameters, learning-rate schedule and fixed stopping epoch selected on development set. Fresh weights; all 25,720 train rows; no official validation evaluation during training. Official validation only after this fixed refit.','device':info,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'data_fingerprints':source['files'],'precision':'BF16 convolutions / FP32 head and loss; TF32 allowed'}
 if mode=='profile':
  config['profile_features']=json.loads((ROOT/'features/train_profile.json').read_text())
  assert config['profile_features']['samples_on_segment']==192
  assert config['profile_features']['inputs']=={k:source['files'][k] for k in ['train_index.csv','train_hm.npy']}
 if not args.resume:
  atomic_json(out/'config.json',config);(out/'source').mkdir(exist_ok=True)
  for f in files:shutil.copy2(f,out/'source'/f.name)
 data=ResidentData('train');train_idx=data.indices('full')
 if mode=='profile':
  profile_meta=json.loads((ROOT/'features/train_profile.json').read_text());assert profile_meta['samples_on_segment']==192
  data.profile=torch.load(ROOT/'features/train_profile.pt',map_location='cuda',weights_only=True)
 model=model_for(mode,width);opt=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=.0001,fused=True);start_epoch=1;history=[]
 if args.resume:
  saved=json.loads((out/'config.json').read_text());assert all(saved[k]==config[k] for k in ('mode','width','batch','epochs','fixed_stop_epoch','lr','seed','selected_dev_run'))
  ck=torch.load(out/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start_epoch=ck['epoch']+1;history=json.loads((out/'history.json').read_text())
 net=torch.compile(model) if selected['compile'] else model;tstart=time.time();nparams=sum(t.numel() for t in model.parameters())
 print(json.dumps({'event':'full_training_start','config':config,'samples':len(train_idx),'parameters':nparams}),flush=True)
 for epoch in range(start_epoch,stop_epoch+1):
  epstart=time.time();rng=np.random.default_rng(args.seed+epoch);order=rng.permutation(train_idx);num_steps=math.ceil(len(order)/batch)
  lr=base_lr*min(1.,epoch/3.)*(.01+.99*(1+math.cos(math.pi*max(0,epoch-3)/max(1,schedule_epochs-3)))/2)
  for group in opt.param_groups:group['lr']=lr
  model.train();train_sum=torch.zeros((),device='cuda');seen=0
  bar=tqdm(range(num_steps),desc=f'{args.run} {epoch}/{stop_epoch}',mininterval=2.,ncols=100)
  for step in bar:
   ii=order[step*batch:(step+1)*batch];x,y,m=data.batch(ii,mode,int(rng.integers(0,8)));opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x);loss=masked_loss(pred,y,m)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,foreach=True,error_if_nonfinite=True);opt.step();train_sum+=loss.detach()*len(ii);seen+=len(ii)
   if step%20==0 or step+1==num_steps:
    avg=float(train_sum)/seen*SCALE*SCALE;bar.set_postfix(mse_db=f'{avg:.2f}',refresh=False)
    atomic_json(out/'progress.json',{'status':'training','epoch':epoch,'fixed_stop_epoch':stop_epoch,'completed_batches':step+1,'total_batches':num_steps,'train_mse_db_batch_weighted':avg,'elapsed_seconds':time.time()-tstart,'gpu_uuid':GPU_UUID,'pid':os.getpid(),'updated_unix':time.time()})
   if stop_requested or (out/'STOP').exists():break
  torch.cuda.synchronize();seconds=time.time()-epstart
  if stop_requested or (out/'STOP').exists():
   checkpoint(out/'interrupted.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'completed_batches':step+1,'config':config});atomic_json(out/'progress.json',{'status':'stopped','epoch':epoch,'completed_batches':step+1,'pid':os.getpid()});break
  record={'epoch':epoch,'lr':lr,'train_mse_db_batch_weighted':float(train_sum)/seen*SCALE*SCALE,'train_seconds':seconds,'training_maps_per_second':seen/seconds}
  history.append(record);atomic_json(out/'history.json',history);checkpoint(out/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':config})
  if epoch%10==0:checkpoint(out/f'epoch_{epoch:03d}.pt',{'model':model.state_dict(),'epoch':epoch,'config':config})
  print('EPOCH_RESULT '+json.dumps(record),flush=True)
 if not stop_requested and not (out/'STOP').exists():
  checkpoint(out/'best.pt',{'model':model.state_dict(),'epoch':stop_epoch,'config':config,'selection':'Fixed stopping epoch selected on development set before refit; not best official-validation checkpoint'})
  result={'run':args.run,'evaluation_split':'not_evaluated','completed_epochs':len(history),'fixed_stop_epoch':stop_epoch,'selected_dev_run':args.selected_dev_run,'goal_met_on_official_validation':False,'official_validation_used_during_training':False,'total_seconds':time.time()-tstart,'parameters':nparams,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9}
  atomic_json(out/'result.json',result);atomic_json(out/'progress.json',{'status':'completed',**result,'pid':os.getpid(),'updated_unix':time.time()});print('FIXED_TRAINING_COMPLETE '+json.dumps(result),flush=True)
except Exception as exc:
 atomic_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'time':time.time(),'pid':os.getpid()});raise
