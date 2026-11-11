import os,time,json,math,hashlib,argparse,random
from pathlib import Path
import numpy as np
import torch
from c21_model_r1 import Model
from cc_core import ROOT,atomic_json,gpu_guard,check_source_unchanged
A=ROOT/'audit/c21_execution_20261101'
def setup(seed):
 gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15,0);torch.set_num_threads(8)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.benchmark=True
 return gpu
def model(geometry):
 n=np.load(A/'normalization.npz')
 return Model(torch.from_numpy(n['mean']),torch.from_numpy(n['std']),geometry=geometry).cuda()
def gather(a,pix):return a.flatten(2).gather(2,pix[:,None].expand(-1,2,-1)).transpose(1,2)
def digest_state(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
@torch.inference_mode()
def evaluate_loaded(net,dat,chunk=1024,keep=False,amp=True):
 net.eval();x,tx,fs,y,m=dat;ss=torch.zeros(2,device='cuda',dtype=torch.float64);cc=torch.zeros_like(ss);maps=[];preds=[]
 for j in range(len(x)):
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):state=net.encode(x[j:j+1],tx[j:j+1])
  p=torch.empty((1,65536,2),device='cuda')
  for lo in range(0,65536,chunk):
   pix=torch.arange(lo,min(lo+chunk,65536),device='cuda')[None]
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):p[:,lo:lo+pix.shape[1]]=net.decode(state,pix,chunk)*20+gather(fs[j:j+1],pix)
  yy=y[j].flatten(1).T[None];mm=m[j].flatten(1).T[None];s=((p.double()-yy.double()).square()*mm).sum((0,1));c=mm.sum((0,1),dtype=torch.float64)
  ss+=s;cc+=c;maps.append({'sse':s.tolist(),'count':c.tolist()})
  if keep:preds.append(p[0].T.reshape(2,256,256).cpu().numpy())
 r=(ss/cc).sqrt().tolist();net.train()
 return {'rmse':r,'score':sum(r)/2,'sse':ss.tolist(),'count':cc.tolist(),'maps':maps},preds
def small(args):
 gpu=setup(args.seed);check_source_unchanged();run=ROOT/'runs'/args.name;run.mkdir(exist_ok=True)
 cfg=vars(args)|{'gpu':gpu,'scope':'four FIT terrains; full valid pixel evaluation; no DEV or officialVAL','gate_both_bands_db':.5,'memory_fraction':.15}
 data=tuple(torch.from_numpy(np.load(A/f'small_{n}.npy')).cuda() for n in ['x','tx','fs','y','m']);x,tx,fs,y,mask=data
 net=model(args.geometry)
 if args.resume:
  ck=torch.load(args.resume,map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);cfg['resume_from_step']=ck['step'];cfg['resume_checkpoint_sha256']=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest();del ck
 cfg['parameters']=sum(p.numel() for p in net.parameters());cfg['trainable_parameters']=sum(p.numel() for p in net.parameters() if p.requires_grad);cfg['initial_state_sha256']=digest_state(net)
 cfg['source_hashes']={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in ['c21_train_r1.py','c21_model_r1.py','c21_model.py','c21_data.py','c21_phase.cpp']}
 atomic_json(run/'config.json',cfg);opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-5)
 gen=torch.Generator(device='cuda').manual_seed(args.seed+1);t0=time.time();best=1e9
 log=(run/'history.jsonl').open('a',buffering=1)
 for step in range(1,args.steps+1):
  lr=args.lr*min(step/50,1)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-50)/max(1,args.steps-50))))
  for pg in opt.param_groups:pg['lr']=lr
  pix=torch.randint(65536,(len(x),args.queries),generator=gen,device='cuda')
  yp=gather(y,pix);mp=gather(mask,pix);base=gather(fs,pix)
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x,tx,pix,args.chunk)*20+base
  loss=(((pred-yp).square()*mp).sum((0,1))/mp.sum((0,1)).clamp_min(1)).mean()/400
  if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss step {step}')
  loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  if step<=3 or step%50==0:
   row={'event':'train','step':step,'seconds':time.time()-t0,'sampled_rmse':float(torch.sqrt(loss*400)),'lr':lr,'grad_norm':float(grad),'peak_GB':torch.cuda.max_memory_allocated()/1e9}
   print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');atomic_json(run/'progress.json',row)
  if step==1 or step%args.eval_every==0 or step==args.steps:
   torch.cuda.synchronize();te=time.time();metrics,_=evaluate_loaded(net,data,args.eval_chunk)
   row={'event':'eval','step':step,'seconds':time.time()-t0,'eval_seconds':time.time()-te,**metrics,'geometry_weights_norm':float(net.geoweight.norm())}
   print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');atomic_json(run/'latest_eval.json',row)
   if row['score']<best:
    best=row['score'];torch.save({'model':net.state_dict(),'step':step,'metrics':row,'config':cfg},run/'best.pt');atomic_json(run/'best.json',row)
   if max(row['rmse'])<=.5:
    atomic_json(run/'complete.json',{'status':'small_fit_gate_passed',**row});break
 else:atomic_json(run/'complete.json',{'status':'small_fit_gate_not_met',**row})
 log.close()
def bench(args):
 gpu=setup(args.seed);x,tx,fs,y,m=tuple(torch.from_numpy(np.load(A/f'small_{n}.npy')).cuda() for n in ['x','tx','fs','y','m']);results=[]
 for geo in [False,True]:
  torch.manual_seed(args.seed);net=model(geo);opt=torch.optim.AdamW(net.parameters(),lr=3e-4)
  pix=torch.randint(65536,(len(x),args.queries),device='cuda');torch.cuda.reset_peak_memory_stats()
  timings=[]
  for i in range(5):
   torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x,tx,pix,args.chunk)*20+gather(fs,pix)
   mm=gather(m,pix);loss=(((p-gather(y,pix)).square()*mm).sum((0,1))/mm.sum((0,1))).mean()/400
   loss.backward();opt.step();torch.cuda.synchronize();timings.append(time.time()-t)
  te=time.time();ev,_=evaluate_loaded(net,(x[:1],tx[:1],fs[:1],y[:1],m[:1]),args.eval_chunk);torch.cuda.synchronize()
  res={'geometry':geo,'parameters':sum(p.numel() for p in net.parameters()),'step_seconds':timings,'dense_map_seconds':time.time()-te,'peak_GB':torch.cuda.max_memory_allocated()/1e9,'geometry_gradient_norm':float(net.geoweight.grad.norm()) if geo else None,'metrics_after5steps':ev}
  results.append(res);print(json.dumps(res),flush=True)
  del net,opt,p,loss;torch.cuda.empty_cache()
 atomic_json(A/'benchmark.json',{'gpu':gpu,'args':vars(args),'results':results})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['small','bench'],default='small');ap.add_argument('--name',default='c21_small_a');ap.add_argument('--geometry',type=int,default=0);ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--queries',type=int,default=2048);ap.add_argument('--chunk',type=int,default=512);ap.add_argument('--eval-chunk',type=int,default=1024);ap.add_argument('--eval-every',type=int,default=250);ap.add_argument('--seed',type=int,default=20261119);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--resume',default='')
 a=ap.parse_args();(small if a.mode=='small' else bench)(a)
