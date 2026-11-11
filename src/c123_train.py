import sys,json,time,hashlib,math,argparse,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from collections import Counter
from src.c105_context import reader,ResidualPanel,digest
from src.c108_train import evaluate
from src.c123_lossless import create
A=R/'audit/c123_lossless'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['native','mean'],required=True);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 old=json.loads((R/'audit/c105_context/protocol.json').read_text())
 plan=json.loads((A/'protocol.json').read_text());batch=plan['batch'];net=create(a.mode);h=digest(net);assert h==plan['initial_state_sha256']
 out=R/'runs'/('c123_'+a.mode);out.mkdir(exist_ok=False);net=net.cuda().to(memory_format=torch.channels_last)
 opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01,fused=True)
 start=time.monotonic();r=reader();train=ResidualPanel(r,old['fit'],'context');monitor=ResidualPanel(r,old['monitor'],'context')
 config=dict(mode=a.mode,GPU=g,parameters=sum(p.numel() for p in net.parameters()),initial_sha=h,fit_maps=len(train.ids),fit_terrains=len(set(train.terrain)),batch=batch,max_epochs=plan['epochs'],official_val_used=False)
 atomic_json(out/'config.json',config)
 init=evaluate(net,monitor,batch,force_base=True);best=float(np.mean(init['rmse']));be=0;history=[]
 atomic_json(out/'initial.json',init);torch.save(dict(state=net.state_dict(),step=0,epoch=0,config=config),out/'best.pt')
 fullfit=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(r.rows[int(i)]['terrain_id'] for i in fullfit);sel=Counter(train.terrain)
 weight=torch.tensor([cnt[t]/sel[t] for t in train.terrain],device='cuda');weight/=weight.mean()
 gen=torch.Generator().manual_seed(123073);steps=math.ceil(len(train.ids)/batch)
 for ep in range(1,plan['epochs']+1):
  net.train();order=torch.randperm(len(train.ids),generator=gen).cuda();ss=torch.zeros(4,device='cuda',dtype=torch.float64)
  for j,ix in enumerate(order.split(batch)):
   step=(ep-1)*steps+j+1
   lr=.0003*min(1.,.1+.9*step/100)*(.15+.85*.5*(1+math.cos(math.pi*(step-1)/(steps*plan['epochs']))))
   for pg in opt.param_groups:pg['lr']=lr
   x,b,y,w=train.batch(ix)
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
   e=torch.where(w,p-y,0.);ww=weight[ix,None,None,None];s=(e.square()*ww).sum((0,2,3));c=(w*ww).sum((0,2,3));loss=(s/c).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   ss+=torch.cat([s.detach(),c])
  val=evaluate(net,monitor,batch);criterion=float(np.mean(val['rmse']))
  if criterion<best:
   best=criterion;be=ep;torch.save(dict(state=net.state_dict(),step=step,epoch=ep,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  rec=dict(epoch=ep,step=step,fit_running_rmse=(ss[:2]/ss[2:]).sqrt().tolist(),monitor=val,gain=(np.array(init['rmse'])-val['rmse']).tolist(),best_epoch=be,seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(rec),flush=True)
  # Per-arm early-stop is conservative and prespecified; compare matched prefixes separately.
  if ep>=4 and ep-be>=3:break
 torch.save(dict(state=net.state_dict(),step=step,epoch=ep,config=config),out/'last.pt')
 atomic_json(out/'result.json',dict(status='complete',best_epoch=be,last_epoch=ep,seconds=time.monotonic()-start,official_val_used=False))
 print('DONE',a.mode,be,ep,flush=True)
if __name__=='__main__':main()
