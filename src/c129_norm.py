"""Matched optimizer continuation: batch ratio vs exact full-FIT normalization."""
import sys,json,time,math,argparse,os,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c127_train import create,reader,Panel,weights,group_batch,evaluate
from src.c105_context import digest
A=R/'audit/c129_fitgap'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['batch','global'],required=True);args=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((R/'audit/c127_joint/protocol.json').read_text());rr=reader();start=time.monotonic()
 train=Panel(rr,plan['fit'],reference=True);monitor=Panel(rr,plan['monitor'],reference=True);weight=weights(rr,train)
 net=create('residual');parent=R/'runs/c127_residual_independent/best.pt';ck=torch.load(parent,map_location='cpu',weights_only=False);assert ck['epoch']==4;net.load_state_dict(ck['state']);initial=digest(net)
 net=net.cuda().to(memory_format=torch.channels_last)
 global_den=(train.w*weight[:,None,None,None]).sum((0,2,3),dtype=torch.float64).float()
 out=R/('runs/c129_norm_'+args.mode);out.mkdir(exist_ok=False)
 config=dict(mode=args.mode,GPU=gpu,parent=str(parent.relative_to(R)),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),initial_sha256=initial,max_epochs=8,min_epochs=4,patience=3,
  batch=16,lr=.0003,weight_decay=.01,input='unchanged56H/TX;no mask input',official_val_used=False,
  data_order='exact first8C127 CPU frozen grouped epochs',normalization='global_den/number_of_batches_in_this_epoch' if args.mode=='global' else 'perbatch weighted valid count')
 atomic_json(out/'config.json',config);atomic_json(out/'indices.json',monitor.ids)
 opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True)
 initial_metric=evaluate(net,monitor,'residual',16);atomic_json(out/'initial.json',initial_metric)
 assert max(abs(a-b) for a,b in zip(initial_metric['rmse'],[2.1420572246029,2.4142969240896393]))<1e-5
 best=sum(initial_metric['rmse'])/2;be=0;history=[];torch.save(dict(state=net.state_dict(),epoch=0,config=config),out/'best.pt')
 nstep=sum(map(len,plan['orders'][:8]));step=0
 for ep,batches in enumerate(plan['orders'][:8],1):
  totals=torch.zeros(4,device='cuda',dtype=torch.float64)
  for j,ix in enumerate(batches):
   step+=1;lr=.0003*min(1.,.1+.9*step/100)*(.15+.85*.5*(1+math.cos(math.pi*(step-1)/nstep)))
   for pg in opt.param_groups:pg['lr']=lr
   idx=torch.tensor(ix,device='cuda');x,_,y,v=train.batch(idx);base=train.reference[idx];_,first,inverse=group_batch(train,ix)
   with torch.autocast('cuda',torch.bfloat16):s,t=net.components(x,first,inverse)
   e=torch.where(v,base+s+t-y,0.);ss=(e.square()*weight[idx,None,None,None]).sum((0,2,3));count=(v*weight[idx,None,None,None]).sum((0,2,3))
   denom=count if args.mode=='batch' else global_den/len(batches)
   loss=(ss/denom.clamp_min(1)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   totals+=torch.cat([ss.detach(),count])
   if (j+1)%80==0:
    rec=dict(epoch=ep,step=j+1,steps=len(batches),seconds=time.monotonic()-start)
    atomic_json(out/'progress.json',rec);print('PROGRESS',json.dumps(rec),flush=True)
  val=evaluate(net,monitor,'residual',16);fit=evaluate(net,train,'residual',16,plan['fit_eval_positions'])
  if np.mean(val['rmse'])<best:
   best=np.mean(val['rmse']);be=ep;torch.save(dict(state=net.state_dict(),epoch=ep,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  rec=dict(epoch=ep,fit_running_weighted_RMSE=(totals[:2]/totals[2:]).sqrt().tolist(),FIT_panel=fit,monitor=val,best_epoch=be,seconds=time.monotonic()-start)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(dict(epoch=ep,monitor=val['rmse'],FIT=fit['rmse'],best_epoch=be,seconds=rec['seconds'])),flush=True)
  torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),epoch=ep,config=config),out/'last.pt')
  if ep>=4 and ep-be>=3:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 val=evaluate(net,monitor,'residual',16,save=out/'monitor_prediction.npy')
 result=dict(status='complete',best_epoch=be,last_epoch=ep,monitor=val,gain_vs_C82=(np.array(val['C82'])-val['rmse']).tolist(),seconds=time.monotonic()-start,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
