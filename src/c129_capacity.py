"""Matched-exposure and same-family width diagnostics."""
import sys,json,time,argparse,hashlib,os,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c127_train import reader,Panel,group_batch,evaluate,create
from src.c105_context import digest
from src.c129_capacity_model import initialize,WideField,expand_state
A=R/'audit/c129_fitgap'
def smoke(g):
 p=json.loads((A/'capacity_protocol.json').read_text());data=Panel(reader(),p['fit16'],reference=True)
 idx=torch.arange(16,device='cuda');x,_,y,v=data.batch(idx);_,first,inverse=group_batch(data,list(range(16)))
 old=create('residual');old.load_state_dict(torch.load(R/'runs/c127_residual_independent/best.pt',map_location='cpu',weights_only=False)['state'])
 wide=WideField(2);expand_state(old,wide);old=old.cuda();wide=wide.cuda()
 torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
 with torch.no_grad():
  a=old(x[:2]);b=wide(x[:2]);err=(a-b).float();diff=dict(max=float(err.abs().max()),rms=float(err.square().mean().sqrt()))
 assert diff['rms']<1e-5 and diff['max']<1e-3,diff
 del old,wide;torch.cuda.empty_cache();torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
 stats={}
 for k in [1,2]:
  net=initialize(k);h=digest(net);torch.save(dict(state=net.state_dict(),hash=h),A/f'capacity_initial_w{k}.pt')
  net=net.cuda().to(memory_format=torch.channels_last)
  with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):delta=net(x,first,inverse)
  assert float(delta.abs().max())==0.
  opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True);tim=[]
  den=v.sum((0,2,3)).float()
  for step in range(8):
   torch.cuda.synchronize();start=time.monotonic()
   with torch.autocast('cuda',torch.bfloat16):pred=data.reference+net(x,first,inverse)
   e=torch.where(v,pred-y,0);loss=(e.square().sum((0,2,3))/den).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss);torch.cuda.synchronize();tim.append(time.monotonic()-start)
  stats[str(k)]=dict(parameters=sum(v.numel() for v in net.parameters()),initial_hash=h,seconds_per_step=float(np.mean(tim[3:])),max_memory_GiB=torch.cuda.max_memory_allocated()/2**30)
  del net,opt;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 atomic_json(A/'capacity_smoke.json',dict(status='passed',GPU=g,width_expansion_replay=diff,widths=stats));print('SMOKE',json.dumps(dict(replay=diff,widths=stats)),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--arm',choices=['n16','n64','w64']);args=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 if args.smoke:smoke(gpu);return
 assert json.loads((A/'capacity_smoke.json').read_text())['status']=='passed'
 plan=json.loads((A/'capacity_protocol.json').read_text());arm=plan['arms'][args.arm];ids=plan['fit'+str(arm['maps'])];rr=reader();start=time.monotonic()
 data=Panel(rr,ids,reference=True);monitor=Panel(rr,plan['monitor'],reference=True)
 net=initialize(arm['width']);ck=torch.load(A/f'capacity_initial_w{arm["width"]}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);assert digest(net)==ck['hash'];net=net.cuda().to(memory_format=torch.channels_last)
 out=R/('runs/c129_capacity_'+args.arm);out.mkdir(exist_ok=False)
 config=dict(arm=args.arm,GPU=gpu,maps=len(ids),width=arm['width'],parameters=sum(v.numel() for v in net.parameters()),initial_hash=ck['hash'],
  batch=16,max_exposures=plan['max_exposures'],lr=plan['lr'],weight_decay=plan['weight_decay'],sample_seed=plan['sample_seed'],FIT_ids=ids,
  source_sha256={str(q.relative_to(R)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),R/'src/c129_capacity_model.py']},official_val_used=False,
  selection='FIT-onlybothbands<.95 after100exposures; no monitor selection')
 atomic_json(out/'config.json',config);atomic_json(out/'indices.json',monitor.ids)
 opt=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'],fused=True)
 initial=evaluate(net,monitor,'residual',16);atomic_json(out/'monitor_initial.json',initial)
 initialfit=evaluate(net,data,'residual',16);atomic_json(out/'fit_initial.json',initialfit)
 den=data.w.sum((0,2,3)).float();batches_per_epoch=math.ceil(len(ids)/16);history=[];step=0
 for ep in range(1,plan['max_exposures']+1):
  order=np.random.default_rng(plan['sample_seed']+ep).permutation(len(ids))
  for ix0 in np.array_split(order,batches_per_epoch):
   ix=ix0.tolist();idx=torch.tensor(ix,device='cuda');x,_,y,v=data.batch(idx);base=data.reference[idx];_,first,inverse=group_batch(data,ix)
   with torch.autocast('cuda',torch.bfloat16):pred=base+net(x,first,inverse)
   e=torch.where(v,pred-y,0);loss=(e.square().sum((0,2,3))/(den/batches_per_epoch)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss);step+=1
  if ep%25==0 or ep in [1,24,100,250,500,1000,1700]:
   fit=evaluate(net,data,'residual',16)
   rec=dict(exposures=ep,updates=step,FIT=fit,seconds=time.monotonic()-start)
   history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec)
   torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),exposures=ep,updates=step,config=config),out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
   if ep in [24,100,250,500,1000,1700]:torch.save(dict(state=net.state_dict(),exposures=ep,updates=step,config=config),out/f'ep{ep}.pt')
   print('FIT',json.dumps(dict(exposures=ep,updates=step,rmse=fit['rmse'],tail=fit['tail_rmse'],seconds=rec['seconds'])),flush=True)
   if ep>=100 and max(fit['rmse'])<.95:break
 fit=evaluate(net,data,'residual',16,save=out/'fit_prediction.npy')
 mon=evaluate(net,monitor,'residual',16,save=out/'monitor_prediction.npy')
 result=dict(status='complete',exposures=ep,updates=step,FIT=fit,monitor=mon,FIT_gate=max(fit['rmse'])<1,
  gain_vs_C82=(np.array(mon['C82'])-mon['rmse']).tolist(),seconds=time.monotonic()-start,official_val_used=False,
  scope='FITcapacity/exposure diagnosis; monitor after FIT-only stop, not candidate expansion')
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
