"""Paired C21 FIT/DEV control. No official validation in this program."""
import time,json,math,hashlib,argparse,random,shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from c21_data import Data,R
from c21_train import setup,model,gather,digest_state,evaluate_loaded
from cc_core import atomic_json,check_source_unchanged
A=R/'audit/c21_execution_20261101'
class Maps(Dataset):
 def __init__(self,ids):self.ids=np.asarray(ids);self.data=None
 def __len__(self):return len(self.ids)
 def __getitem__(self,j):
  if self.data is None:self.data=Data()
  i=int(self.ids[j]);return (i,*self.data.sample(i))
def gpu_batch(batch):
 return tuple(t.cuda(non_blocking=True) for t in batch[1:])
def loader(ids,batch=8,workers=4):
 return DataLoader(Maps(ids),batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=False,prefetch_factor=2 if workers else None)
def combine(rows):
 ss=np.array([r['sse'] for r in rows]).sum(0);cc=np.array([r['count'] for r in rows]).sum(0);rm=np.sqrt(ss/cc)
 return {'rmse':rm.tolist(),'score':float(rm.mean()),'sse':ss.tolist(),'count':cc.tolist(),'maps':len(rows)}
@torch.inference_mode()
def monitor(nets,ids,pixels):
 rows=[[],[]]
 for batch in loader(ids,4):
  idx=batch[0].tolist();x,tx,fs,y,m=gpu_batch(batch);pix=torch.from_numpy(pixels[:len(idx)]).cuda()
  yy=gather(y,pix);mm=gather(m,pix);base=gather(fs,pix)
  for k,net in enumerate(nets):
   net.eval()
   with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x,tx,pix,512)*20+base
   ss=((p.double()-yy.double()).square()*mm).sum(1).cpu().numpy();cc=mm.sum(1,dtype=torch.float64).cpu().numpy()
   for i,s,c in zip(idx,ss,cc):rows[k].append({'id':i,'sse':s.tolist(),'count':c.tolist()})
   net.train()
 return [combine(r) for r in rows]
@torch.inference_mode()
def evaluate_batched(net,dat,chunk=1024):
 net.eval();x,tx,fs,y,m=dat;B=len(x)
 with torch.autocast('cuda',dtype=torch.bfloat16):state=net.encode(x,tx)
 p=torch.empty((B,65536,2),device='cuda')
 for lo in range(0,65536,chunk):
  pix=torch.arange(lo,min(lo+chunk,65536),device='cuda')[None].expand(B,-1)
  with torch.autocast('cuda',dtype=torch.bfloat16):p[:,lo:lo+pix.shape[1]]=net.decode(state,pix,chunk)*20+gather(fs,pix)
 yy=y.flatten(2).transpose(1,2);mm=m.flatten(2).transpose(1,2)
 ss=((p.double()-yy.double()).square()*mm).sum(1);cc=mm.sum(1,dtype=torch.float64)
 rows=[{'sse':s.tolist(),'count':c.tolist()} for s,c in zip(ss,cc)]
 pred=p.transpose(1,2).reshape(B,2,256,256).cpu().numpy()
 net.train();return {**combine(rows),'map_rows':rows},pred
@torch.inference_mode()
def full_dev(net,ids,dest):
 dest.mkdir(exist_ok=True);net.eval();predfile=np.lib.format.open_memmap(dest/'pred.npy',mode='w+',dtype=np.float32,shape=(len(ids),2,256,256));rows=[];zone_rows=[];t0=time.time();at=0
 data=Data()
 for batch in loader(ids,4):
  idx=batch[0].tolist();dat=gpu_batch(batch);metrics,preds=evaluate_batched(net,dat,1024)
  xx=dat[0];emax=2*torch.sinh(xx[:,31:39].amax(1)*3)
  zones=[emax<-.25,emax.abs()<=.25,emax>.25]
  for j,(i,p,mc) in enumerate(zip(idx,preds,metrics['map_rows'])):
   predfile[at+j]=p;rows.append({'id':i,'terrain_id':data.rows[i]['terrain_id'],**mc})
   err=(torch.from_numpy(p).cuda().double()-dat[3][j].double()).square();rr={'id':i}
   for zn,z in zip(['clear','boundary','blocked'],zones):
    mm=dat[4][j]*z[j][None];rr[zn]={'sse':(err*mm).sum((1,2)).tolist(),'count':mm.sum((1,2)).tolist()}
   zone_rows.append(rr)
  at+=len(idx)
  if at%128==0:print(json.dumps({'event':'full_dev','directory':str(dest),'maps':at,'seconds':time.time()-t0}),flush=True)
 predfile.flush();out=combine(rows);out.update({'scope':'complete terrain-disjoint DEV; official scoring formula, not officialVAL','seconds':time.time()-t0,'zones_definition':'floor-TX profile max elevation: clear < -0.25 m, boundary within +/-0.25 m, blocked >0.25 m','zones':{z:combine([r[z] for r in zone_rows]) for z in ['clear','boundary','blocked']}})
 atomic_json(dest/'metrics.json',out);atomic_json(dest/'per_map.json',rows);atomic_json(dest/'zones_per_map.json',zone_rows);atomic_json(dest/'ids.json',list(map(int,ids)));net.train();return out
def main(args):
 gpu=setup(args.seed);check_source_unchanged();run=R/'runs'/args.name;run.mkdir(exist_ok=True)
 for name in ['c21_small_a_refine_s20261129','c21_small_b_refine_s20261129']:
  f=R/'runs'/name/'complete.json';r=json.loads(f.read_text());assert r['status']=='small_fit_gate_passed',(name,r['status'])
 assert (R/'features/c21_phase/complete.json').exists(),'Full input cache must be ready'
 data=Data();rng=np.random.default_rng(args.seed);order=[]
 while len(order)*len(data.ids['fit'])<args.steps*args.batch:
  ids=data.ids['fit'].copy();rng.shuffle(ids);order.append(ids)
 order=np.concatenate(order)[:args.steps*args.batch]
 meta=json.loads((R/'features/c17_spatial/metadata.json').read_text());fit_panel=meta['ids']['fit'][:128];dev_panel=meta['ids']['dev']
 pixels=np.random.default_rng(args.seed+2).choice(65536,size=8192,replace=False).astype(np.int64);pixels=np.tile(pixels,(8,1))
 np.save(run/'map_order.npy',order);np.save(run/'monitor_pixels.npy',pixels[0])
 cfg=vars(args)|{'gpu':gpu,'source_hashes':{n:hashlib.sha256((R/'src'/n).read_bytes()).hexdigest() for n in ['c21_pair.py','c21_model.py','c21_data.py','c21_phase.cpp','c21_train.py']},'fit_maps':len(data.ids['fit']),'dev_maps':len(data.ids['dev']),'fit_panel':fit_panel,'dev_panel':dev_panel,'monitor_scope':'128 FIT and128 DEV maps, fixed8192 uniform pixels each; best checkpoints selected by mean of per-band pooled masked RMSE','primary_final_scope':'all2562 DEV maps, complete valid pixels','map_order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),'input_channels':45,'geometry_parameter_difference':128,'weight_decay':1e-5,'memory_fraction':.15,'promotion_threshold':'both complete DEV band RMSE at least 0.20 dB below C10, paired terrain bootstrap lower95%CI for error reduction above zero; then full refit and officialVAL','officialVAL_used':False}
 atomic_json(run/'config.json',cfg)
 (run/'source').mkdir(exist_ok=True)
 for name in cfg['source_hashes']:shutil.copyfile(R/'src'/name,run/'source'/name)
 nets=[];opts=[];initial=[]
 for geo in [False,True]:
  torch.manual_seed(args.seed);net=model(geo);initial.append(digest_state(net));nets.append(net);opts.append(torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-5))
 assert initial[0]==initial[1];atomic_json(run/'initialization.json',{'sha256':initial,'all_parameters_equal_at_start':True,'total_parameters':[sum(p.numel() for p in n.parameters()) for n in nets],'trainable_parameters':[sum(p.numel() for p in n.parameters() if p.requires_grad) for n in nets]})
 gen=torch.Generator(device='cuda').manual_seed(args.seed+1);best=[1e9,1e9];t0=time.time();log=(run/'history.jsonl').open('a',buffering=1);elapsed_train=[0.,0.]
 for step,batch in enumerate(loader(order,args.batch,args.workers),1):
  dat=gpu_batch(batch);x,tx,fs,y,mask=dat;pix=torch.randint(65536,(len(x),args.queries),generator=gen,device='cuda')
  yy=gather(y,pix);mm=gather(mask,pix);base=gather(fs,pix);losses=[]
  lr=args.lr*min(step/200,1)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-200)/max(1,args.steps-200))))
  for k,(net,opt) in enumerate(zip(nets,opts)):
   for pg in opt.param_groups:pg['lr']=lr
   opt.zero_grad(set_to_none=True);t=time.time()
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x,tx,pix,512)*20+base
   loss=(((pred-yy).square()*mm).sum((0,1))/mm.sum((0,1)).clamp_min(1)).mean()/400
   if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss step{step} model{k}')
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(torch.sqrt(loss*400)));elapsed_train[k]+=time.time()-t
  if step%100==0:
   row={'event':'train','step':step,'seconds':time.time()-t0,'sampled_rmse_A_B':losses,'lr':lr,'maps_seen_per_model':step*args.batch,'peak_GB':torch.cuda.max_memory_allocated()/1e9}
   print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');atomic_json(run/'progress.json',row)
  if step%args.eval_every==0 or step==args.steps:
   fit=monitor(nets,fit_panel,pixels);dev=monitor(nets,dev_panel,pixels);row={'event':'monitor','step':step,'seconds':time.time()-t0,'FIT':fit,'DEV':dev,'geometry_weight_norm':float(nets[1].geoweight.norm())}
   print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');atomic_json(run/'latest_eval.json',row)
   for k,net in enumerate(nets):
    label=['A','B'][k];ck={'model':net.state_dict(),'optimizer':opts[k].state_dict(),'step':step,'metrics':row,'config':cfg};torch.save(ck,run/f'last_{label}.pt')
    if dev[k]['score']<best[k]:
     best[k]=dev[k]['score'];shutil.copyfile(run/f'last_{label}.pt',run/f'best_{label}.pt');atomic_json(run/f'best_{label}.json',{'step':step,'FIT':fit[k],'DEV':dev[k]})
 atomic_json(run/'training_complete.json',{'steps_each':args.steps,'seconds':time.time()-t0,'active_train_seconds_A_B':elapsed_train})
 del opts;torch.cuda.empty_cache();final={}
 for k,net in enumerate(nets):
  label=['A','B'][k];ck=torch.load(run/f'best_{label}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);step=ck['step'];del ck
  final[label]={'selected_step':step,**full_dev(net,data.ids['dev'],run/f'dev_{label}')}
 atomic_json(run/'complete.json',{'status':'paired_control_complete','steps_each':args.steps,'seconds':time.time()-t0,'DEV':final})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='c21_pair_s20261123');ap.add_argument('--seed',type=int,default=20261123);ap.add_argument('--steps',type=int,default=18000);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--queries',type=int,default=2048);ap.add_argument('--lr',type=float,default=.0005);ap.add_argument('--eval-every',type=int,default=3000);ap.add_argument('--workers',type=int,default=8);main(ap.parse_args())
