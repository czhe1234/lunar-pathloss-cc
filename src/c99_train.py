"""Matched local/global axial field pilot, using FIT only and the existing128-terrain monitor."""
import argparse,csv,hashlib,json,math,os,sys,time
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c99_axial import AxialField
from src.c99_data import Reader,Panel
@torch.inference_mode()
def evaluate(net,panel,batch):
 net.eval();rows=[];bases=[]
 for start in range(0,len(panel.ids),batch):
  sl=slice(start,start+batch);x,p0,y,w=panel.batch(sl)
  with torch.autocast('cuda',torch.bfloat16):p=p0+net(x).float()
  se=torch.where(w,(p-y).square(),0.).sum((2,3),dtype=torch.float64);cc=w.sum((2,3))
  rows.append(torch.cat([se,cc],1).cpu().numpy())
  if panel.reference is not None:
   er=torch.where(w,(panel.reference[sl]-y).square(),0.).sum((2,3),dtype=torch.float64)
   bases.append(torch.cat([er,cc],1).cpu().numpy())
 net.train();rows=np.concatenate(rows);a=rows.sum(0)
 base=np.concatenate(bases) if bases else None
 return dict(rmse=np.sqrt(a[:2]/a[2:]).tolist(),sse=a[:2].tolist(),count=a[2:].tolist()),rows,base
def statehash(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['local','global'],required=True)
 ap.add_argument('--batch',type=int,default=16);ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.manual_seed(99041);np.random.seed(99041)
 torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('high')
 plan=json.loads((R/'audit/c99_axial/protocol.json').read_text())
 assert a.epochs==plan['max_epochs'] and a.batch==plan['batch']
 out=R/'runs'/('c99_'+a.mode);out.mkdir(exist_ok=a.resume)
 net=AxialField(a.mode=='global');initial_hash=statehash(net);parameters=sum(p.numel() for p in net.parameters())
 assert initial_hash==plan['initial_state_sha256']
 net=net.cuda().to(memory_format=torch.channels_last)
 opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.01,fused=True)
 reader=Reader();t=time.monotonic()
 print('load',a.mode,len(plan['fit']),len(plan['monitor']),'freeGiB',torch.cuda.mem_get_info()[0]/2**30,flush=True)
 train=Panel(reader,plan['fit']);monitor=Panel(reader,plan['monitor'],reference=True)
 fullfit=np.load(R/'features/c38_dense/fit_map_indices.npy');all_count=Counter(reader.rows[int(i)]['terrain_id'] for i in fullfit);sel=Counter(train.terrain)
 mapweight=torch.tensor([all_count[t]/sel[t] for t in train.terrain],device='cuda');mapweight/=mapweight.mean()
 config=dict(**vars(a),gpu=gpu,parameters=parameters,initial_state_sha256=initial_hash,seed=99041,
  data_order_seed=99073,fit_maps=len(train.ids),fit_terrains=len(sel),monitor_maps=len(monitor.ids),input_channels=56,
  input_only='H,TX and existing45H/TXcompact geometry channels; no C82 output or target/mask inputs',
  prior='Same H/TX-only clipped native prior as C73; all labels float32; separate band masks in loss/score',
  loss='TX-map inclusion weighted maskedMSE mean across bands /25',optimizer='AdamW,lr.0008,wd.01,200updatewarmup,cosine12epochs,minfactor.12,clip2',
  selection='minimum jointmean of monitor bandRMSE, including initialzero correction; patience3epochs; no confirmation labels',
  new_train_count_used_in_loss_only=True,official_val_used=False)
 atomic_json(out/'config.json',config)
 gen=torch.Generator().manual_seed(99073);history=[];begin=0
 if a.resume:
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
  net.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);gen.set_state(ck['order']);begin=ck['epoch'];history=json.loads((out/'history.json').read_text())
  best_ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);best=float(best_ck['criterion']);best_ep=best_ck['epoch']
  initial=json.loads((out/'initial.json').read_text())
 else:
  initial,rr,bb=evaluate(net,monitor,a.batch)
  atomic_json(out/'initial.json',initial);np.save(out/'monitor_c82_by_map.npy',bb)
  z=bb.sum(0);ref=np.sqrt(z[:2]/z[2:]);atomic_json(out/'C82_reference.json',dict(rmse=ref.tolist(),sse=z[:2].tolist(),count=z[2:].tolist()))
  best=float(np.mean(initial['rmse']));best_ep=0
  torch.save(dict(state=net.state_dict(),epoch=0,criterion=best,config=config),out/'best.pt');np.save(out/'monitor_best_by_map.npy',rr)
  print('INITIAL',a.mode,initial['rmse'],'C82',ref.tolist(),'loaded_seconds',round(time.monotonic()-t,1),flush=True)
 ref=np.array(json.loads((out/'C82_reference.json').read_text())['rmse'])
 n=len(train.ids);steps=math.ceil(n/a.batch)
 for ep in range(begin,a.epochs):
  net.train();total=torch.zeros(4,device='cuda',dtype=torch.float64)
  order=torch.randperm(n,generator=gen).to('cuda')
  for step,ix in enumerate(order.split(a.batch)):
   progress=ep+(step+1)/steps;warm=min(1.,.1+.9*(ep*steps+step+1)/200)
   lr=.0008*warm*(.12+.88*.5*(1+math.cos(math.pi*progress/a.epochs)))
   for group in opt.param_groups:group['lr']=lr
   x,p0,y,w=train.batch(ix)
   with torch.autocast('cuda',torch.bfloat16):p=p0+net(x).float()
   e=torch.where(w,p-y,0.);weight=mapweight[ix,None,None,None];se=(e.square()*weight).sum((0,2,3));cc=(w*weight).sum((0,2,3))
   loss=(se/cc.clamp_min(1)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
   assert torch.isfinite(loss),f'nonfinite loss epoch{ep+1}step{step+1}'
   total+=torch.cat([se.detach(),cc])
   if (step+1)%100==0:
    rec=dict(epoch=ep+1,step=step+1,total_steps=steps,fit_running_rmse=(total[:2]/total[2:]).sqrt().tolist(),seconds=round(time.monotonic()-t,1))
    atomic_json(out/'progress.json',rec);print('PROGRESS',a.mode,rec,flush=True)
  val,rr,_=evaluate(net,monitor,a.batch);criterion=float(np.mean(val['rmse']))
  if criterion<best:
   best=criterion;best_ep=ep+1;torch.save(dict(state=net.state_dict(),epoch=best_ep,criterion=best,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt');np.save(out/'monitor_best_by_map.npy',rr)
  rec=dict(epoch=ep+1,fit_rmse=(total[:2]/total[2:]).sqrt().tolist(),monitor=val,gain_vs_C82=(ref-val['rmse']).tolist(),
   best_epoch=best_ep,lr=lr,seconds=round(time.monotonic()-t,1),peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('EPOCH',a.mode,json.dumps(rec),flush=True)
  torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),order=gen.get_state(),epoch=ep+1),out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
  if ep+1-best_ep>=3:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);val,rr,_=evaluate(net,monitor,a.batch)
 np.save(out/'monitor_best_by_map.npy',rr)
 result=dict(status='complete',best_epoch=best_ep,last_epoch=ep+1,monitor=val,gain_vs_C82=(ref-val['rmse']).tolist(),
   seconds=time.monotonic()-t,parameters=parameters,gpu=gpu,confirmation_scored=False,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',a.mode,json.dumps(result),flush=True)
if __name__=='__main__':main()
