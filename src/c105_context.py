"""End-to-end native C82 residual learning; C82 image context versus constant channels."""
import sys,hashlib,json,math,time,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c99_axial import AxialField
from src.c99_data import Reader,Panel
from src.c3_geometry import basic_inputs
A=R/'audit/c105_context'
class ResidualPanel(Panel):
 def __init__(self,reader,ids,mode):
  super().__init__(reader,ids,reference=True);self.mode=mode
 def batch(self,ix):
  x,_,y,w=super().batch(ix);base=self.reference[ix]
  if self.mode=='context':
   _,fs=basic_inputs(self.heights[self.hidx[ix]],self.tr[ix],self.tc[ix])
   extra=(base-fs)/20
  else:extra=torch.zeros_like(base)
  return torch.cat([x,extra],1).contiguous(memory_format=torch.channels_last),base,y,w
def reader():
 r=Reader()
 meta=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text())
 fit=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
 class Joined:
  def __getitem__(self,k):
   source,j=k
   return fit[j] if source=='fit' else r.oldbase[j]
 r.oldbase=r.c82;r.c82=Joined()
 r.c82index={i:('dev',j) for i,j in r.c82index.items()}
 for j,i in enumerate(meta):
  assert i not in r.c82index
  r.c82index[i]=('fit',j)
 return r
def create():
 torch.manual_seed(105041);n=AxialField(True,channels=58)
 ck=torch.load(R/'runs/c99_global/best.pt',map_location='cpu',weights_only=False);assert ck['epoch']==11
 old=ck['state'];state=n.state_dict()
 for k,v in state.items():
  if k in ['stem.weight','film1.weight','film2.weight']:
   v.zero_();v[:,:56]=old[k]
  elif k in ['head.weight','head.bias']:v.zero_()
  else:v.copy_(old[k])
 n.load_state_dict(state);return n
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
@torch.inference_mode()
def score(net,panel,batch,force_base=False):
 net.eval();rows=[];base_rows=[]
 for start in range(0,len(panel.ids),batch):
  sl=slice(start,start+batch);x,p0,y,w=panel.batch(sl)
  if force_base:p=p0
  else:
   with torch.autocast('cuda',torch.bfloat16):p=p0+net(x).float()
  ss=torch.where(w,(p-y).square(),0.).sum((2,3),dtype=torch.float64);cc=w.sum((2,3))
  bss=torch.where(w,(p0-y).square(),0.).sum((2,3),dtype=torch.float64)
  rows.append(torch.cat([ss,cc],1).cpu().numpy());base_rows.append(torch.cat([bss,cc],1).cpu().numpy())
 rows=np.concatenate(rows);br=np.concatenate(base_rows);z=rows.sum(0);bz=br.sum(0);net.train()
 return dict(rmse=np.sqrt(z[:2]/z[2:]).tolist(),sse=z[:2].tolist(),count=z[2:].tolist(),C82=np.sqrt(bz[:2]/bz[2:]).tolist()),rows
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['local','context'],default='context');ap.add_argument('--small',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());n=create();initial_hash=digest(n);assert initial_hash==plan['initial_state_sha256']
 net=n.cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True)
 t=time.monotonic();r=reader();ids=plan['fit'][:16] if a.small else plan['fit']
 train=ResidualPanel(r,ids,a.mode)
 if a.small:
  baseline,_=score(net,train,8,force_base=True);hist=[];gen=torch.Generator().manual_seed(105073)
  for step in range(1,201):
   ix=torch.randperm(len(ids),generator=gen)[:8].cuda();x,b,y,w=train.batch(ix)
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
   e=torch.where(w,p-y,0.);loss=(e.square().sum((0,2,3))/w.sum((0,2,3)).clamp_min(1)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   if step%50==0:
    v,_=score(net,train,8);hist.append(dict(step=step,fit=v['rmse'],seconds=time.monotonic()-t));print(hist[-1],flush=True)
  atomic_json(A/'small.json',dict(initial=baseline,history=hist,GPU=gpu,fit_maps=16,interpretation='Capacity check only, these16maps remain FIT; no selection on confirmation.'))
  return
 monitor=ResidualPanel(r,plan['monitor'],a.mode);out=R/'runs'/('c105_'+a.mode);out.mkdir(exist_ok=False)
 config=dict(mode=a.mode,GPU=gpu,parameters=sum(p.numel() for p in net.parameters()),initial_state_sha256=initial_hash,
  epochs=plan['epochs'],batch=plan['batch'],fit_maps=len(ids),fit_terrains=len(set(train.terrain)),monitor_maps=len(monitor.ids),
  source='C99global epoch11 feature extractor; oldhead discarded; allparameters trainable; both arms start exactC82.',
  input='56rawH/TXgeometry plus2C82excess fullimage incontext, constant2channels inlocal. C82prior inboth.',
  supervision='allnativevalidTRAINpixels,float32labels;maskedperbandMSE/25;mapinclusionweights;no officialVAL',
  selection='minfull128monitorjointmean,includingC82;patience2epochs;freeze then same484confirmation',official_val_used=False)
 atomic_json(out/'config.json',config);initial,rr=score(net,monitor,plan['batch']);atomic_json(out/'initial.json',initial)
 assert np.max(np.abs(np.array(initial['rmse'])-initial['C82']))<1e-10
 np.save(out/'monitor_initial.npy',rr);best=float(np.mean(initial['rmse']));bestep=0;history=[]
 torch.save(dict(state=net.state_dict(),epoch=0,config=config),out/'best.pt')
 from collections import Counter
 fullfit=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(r.rows[int(i)]['terrain_id'] for i in fullfit);sel=Counter(train.terrain)
 weights=torch.tensor([cnt[t]/sel[t] for t in train.terrain],device='cuda');weights/=weights.mean()
 gen=torch.Generator().manual_seed(105073);steps=math.ceil(len(ids)/plan['batch'])
 print('INITIAL',a.mode,initial,'loaded',time.monotonic()-t,flush=True)
 for ep in range(1,plan['epochs']+1):
  order=torch.randperm(len(ids),generator=gen).cuda();ss=torch.zeros(4,device='cuda',dtype=torch.float64)
  for j,ix in enumerate(order.split(plan['batch'])):
   warm=min(1.,.1+.9*((ep-1)*steps+j+1)/100)
   lr=.0004*warm*(.15+.85*.5*(1+math.cos(math.pi*((ep-1)+(j+1)/steps)/plan['epochs'])))
   for pg in opt.param_groups:pg['lr']=lr
   x,b,y,w=train.batch(ix)
   with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
   err=torch.where(w,p-y,0.);ww=weights[ix,None,None,None];se=(err.square()*ww).sum((0,2,3));cc=(w*ww).sum((0,2,3));loss=(se/cc.clamp_min(1)).mean()/25
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   ss+=torch.cat([se.detach(),cc])
  val,rr=score(net,monitor,plan['batch']);criterion=float(np.mean(val['rmse']))
  if criterion<best:
   best=criterion;bestep=ep;torch.save(dict(state=net.state_dict(),epoch=ep,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt');np.save(out/'monitor_best.npy',rr)
  rec=dict(epoch=ep,fit_running_rmse=(ss[:2]/ss[2:]).sqrt().tolist(),monitor=val,gain_vs_C82=(np.array(initial['rmse'])-val['rmse']).tolist(),
   best_epoch=bestep,lr=lr,seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(rec),flush=True)
  if ep-bestep>=2:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 val,rr=score(net,monitor,plan['batch'],force_base=bestep==0)
 atomic_json(out/'result.json',dict(status='complete',best_epoch=bestep,last_epoch=ep,monitor=val,
  gain_vs_C82=(np.array(initial['rmse'])-val['rmse']).tolist(),seconds=time.monotonic()-t,official_val_used=False))
 print('DONE',a.mode,bestep,val,flush=True)
if __name__=='__main__':main()
