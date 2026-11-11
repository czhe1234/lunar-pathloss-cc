"""Controlled16-map fitting: duration,weight decay,learning-rate schedule, then heldout."""
import argparse,sys,time,math,json,os,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c105_context import create,reader,ResidualPanel,digest
A=R/'audit/c106_fit'
@torch.inference_mode()
def evaluate(net,x,base,y,mask,batch=8,save=None):
 net.eval();pred=[]
 for start in range(0,len(x),batch):
  with torch.autocast('cuda',torch.bfloat16):p=base[start:start+batch]+net(x[start:start+batch]).float()
  pred.append(p)
 pred=torch.cat(pred);error=torch.where(mask,pred-y,0.)
 ss=error.square().sum((2,3),dtype=torch.float64);count=mask.sum((2,3));rmse=(ss.sum(0)/count.sum(0)).sqrt();per=(ss/count.clamp_min(1)).sqrt()
 residual=y-base;gs=[];gc=[];gp=[]
 for k in [0,1,2]:
  g=mask&((residual.abs()<5) if k==0 else (residual>=5) if k==1 else (residual<=-5))
  gs.append(torch.where(g,error.square(),0.).sum((0,2,3),dtype=torch.float64));gc.append(g.sum((0,2,3)))
  gp.append(torch.where(g,pred-y,0.).sum((0,2,3),dtype=torch.float64))
 gs=torch.stack(gs);gc=torch.stack(gc);gp=torch.stack(gp)
 result=dict(rmse=rmse.tolist(),sse=ss.sum(0).tolist(),count=count.sum(0).tolist(),
  per_map_band_rmse=per.tolist(),max_map_band_rmse=per.max(0).values.tolist(),
  residual_group_rmse=(gs/gc.clamp_min(1)).sqrt().tolist(),residual_group_bias=(gp/gc.clamp_min(1)).tolist(),
  residual_group_count=gc.tolist(),passed=bool((rmse<.95).all() and (per<.98).all()))
 if save is not None:np.save(save,pred.cpu().numpy())
 net.train();return result
def loss_value(net,x,b,y,w):
 with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
 e=torch.where(w,p-y,0.);return (e.square().sum((0,2,3))/w.sum((0,2,3)).clamp_min(1)).mean()/25
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['original','no_decay','lr_decay'],default='original');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 plan=json.loads((A/'protocol.json').read_text());net=create();h=digest(net);assert h==plan['initial_state_sha256']
 net=net.cuda().to(memory_format=torch.channels_last);wd=0. if a.mode=='no_decay' else .01
 opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=wd,fused=True)
 t=time.monotonic();panel=ResidualPanel(reader(),plan['fit16'],'context')
 x,b,y,w=panel.batch(slice(0,16));assert x.shape==(16,58,256,256)
 assert bool(torch.isfinite(x).all() and torch.isfinite(b).all() and torch.isfinite(y).all())
 # Cache a literal copy of the same H/TX/C82 forward inputs, without labels/masks.
 for sl in [slice(0,8),slice(8,16)]:
  xx,bb,yy,ww=panel.batch(sl);assert torch.equal(x[sl],xx) and torch.equal(b[sl],bb) and torch.equal(y[sl],yy) and torch.equal(w[sl],ww)
 if a.smoke:
  gen=torch.Generator().manual_seed(105073);timing=[]
  for step in range(12):
   ix=torch.randperm(16,generator=gen)[:8].cuda();torch.cuda.synchronize();t0=time.monotonic()
   opt.zero_grad(set_to_none=True);loss=loss_value(net,x[ix],b[ix],y[ix],w[ix]);loss.backward();assert torch.isfinite(loss)
   torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();torch.cuda.synchronize();timing.append(time.monotonic()-t0)
  out=dict(GPU=gpu,batch=8,seconds_per_update=float(np.mean(timing[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30,finite_loss=float(loss))
  atomic_json(A/'smoke.json',out);print(json.dumps(out),flush=True);return
 out=R/'runs'/('c106_'+a.mode);out.mkdir(exist_ok=False)
 config=dict(mode=a.mode,GPU=gpu,initial_state_sha256=h,parameters=sum(p.numel() for p in net.parameters()),fit16=plan['fit16'],
  batch=8,max_steps=plan['max_steps'],seed=105041,sample_seed=105073,weight_decay=wd,initial_lr=.0004,
  schedule='constant' if a.mode!='lr_decay' else 'constant200,then cosine to.00002at4000',
  inherited='C105context initialization,not C105small optimizerstate; no saved small checkpoint existed',
  input='same58H/TXgeometry+C82excess channels;no label,mask,IDinputs;no permapparameters',official_val_used=False)
 atomic_json(out/'config.json',config);initial=evaluate(net,x,b,y,w);atomic_json(out/'initial.json',initial)
 assert np.max(np.abs(np.array(initial['rmse'])-np.array(plan['baseline16_rmse'])))<1e-5
 print('INITIAL',a.mode,initial['rmse'],flush=True)
 history=[];gen=torch.Generator().manual_seed(105073);best=float(np.mean(initial['rmse']));beststep=0;passed=False
 torch.save(dict(state=net.state_dict(),step=0,config=config),out/'best.pt')
 for step in range(1,plan['max_steps']+1):
  ix=torch.randperm(16,generator=gen)[:8].cuda()
  lr=.0004
  if a.mode=='lr_decay' and step>200:
   lr=.00002+.00038*.5*(1+math.cos(math.pi*(step-200)/(plan['max_steps']-200)))
  for pg in opt.param_groups:pg['lr']=lr
  opt.zero_grad(set_to_none=True);loss=loss_value(net,x[ix],b[ix],y[ix],w[ix]);loss.backward()
  grad=torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss) and torch.isfinite(grad)
  if step%100==0 or step==plan['max_steps']:
   val=evaluate(net,x,b,y,w);criterion=float(np.mean(val['rmse']))
   if criterion<best:
    best=criterion;beststep=step
    torch.save(dict(state=net.state_dict(),step=step,config=config),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   rec=dict(step=step,fit=val,loss=float(loss),gradient_norm_before_clip=float(grad),lr=lr,seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30,best_step=beststep)
   history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec)
   print('FIT',a.mode,json.dumps(rec),flush=True)
   if val['passed']:
    passed=True;torch.save(dict(state=net.state_dict(),step=step,config=config),out/'first_pass.pt');break
 ck=torch.load(out/'first_pass.pt' if passed else out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 final=evaluate(net,x,b,y,w,save=out/'fit_prediction.npy')
 np.savez(out/'fit_truth.npz',target=y.cpu().numpy(),mask=w.cpu().numpy(),base=b.cpu().numpy(),mapid=np.array(plan['fit16']))
 result=dict(status='complete',passed=passed,selected_step=ck['step'],last_step=step,fit=final,seconds=time.monotonic()-t,
  selected_checkpoint='first_pass.pt' if passed else 'best.pt',official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',a.mode,json.dumps(result),flush=True)
if __name__=='__main__':main()
