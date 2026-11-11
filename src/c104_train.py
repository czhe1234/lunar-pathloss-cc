"""C82 signed-tail state/conditional-amplitude learning versus importance-sampled MSE."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from src.c34_model import Model
from src.c103_train import Data,digest
A=R/'audit/c104_tail'
NORM=R/'audit/c103_terrain_structure/normalization.npz'
def create(mode,mean,std,prior,proposal,class_mean):
 torch.manual_seed(104041)
 net=Model('periodic',8,512,mean,std,torch.zeros(235,17),torch.zeros(2))
 net.head=nn.Parameter(torch.zeros(8,512,12));net.bias=nn.Parameter(torch.zeros(8,12))
 with torch.no_grad():
  net.embedding.frequencies.copy_(torch.tensor([.0625,.125,.25,.5,1.,2.,4.,8.])[None].expand(235,-1))
  if mode=='state':
   bias=net.bias.view(8,2,6)
   bias[:,:,:3]=torch.log(proposal)[None]
   mu0=-(prior[:,1]*class_mean[:,1]+prior[:,2]*class_mean[:,2])/prior[:,0]
   assert (mu0.abs()<5).all()
   bias[:,:,3]=torch.atanh(mu0/5)[None]
   bias[:,:,4]=torch.log(torch.expm1(class_mean[:,1]-5))[None]
   bias[:,:,5]=torch.log(torch.expm1(-class_mean[:,2]-5))[None]
 return net
def decoded(raw,mode,prior,proposal):
 raw=raw.view(-1,8,2,6)
 if mode=='mse':return raw[:,:,:,0]*5,None,None
 logits=raw[:,:,:,:3]
 mu=torch.stack((5*torch.tanh(raw[:,:,:,3]),5+F.softplus(raw[:,:,:,4]),-5-F.softplus(raw[:,:,:,5])),-1)
 p=torch.softmax(logits+torch.log(prior/proposal)[None,None],-1)
 return (p*mu).sum(-1),logits,mu
class Sampler:
 def __init__(self):
  z=np.load(A/'sampler.npz')
  self.prior=torch.tensor(z['prior'],device='cuda',dtype=torch.float32)
  self.proposal=torch.tensor(z['proposal'],device='cuda',dtype=torch.float32)
  self.means=torch.tensor(z['class_mean'],device='cuda',dtype=torch.float32)
  self.ix={(b,c):torch.tensor(z[f'index{b}{c}'],device='cuda',dtype=torch.int64) for b in range(2) for c in range(3)}
  self.cdf={(b,c):torch.tensor(z[f'cdf{b}{c}'],device='cuda',dtype=torch.float64) for b in range(2) for c in range(3)}
 def sample(self,n,gen):
  assert n%2==0
  band=torch.arange(2,device='cuda').repeat_interleave(n//2)
  cls=torch.cat([torch.multinomial(self.proposal[b],n//2,replacement=True,generator=gen) for b in range(2)])
  u=torch.rand(n,device='cuda',generator=gen,dtype=torch.float64)
  ix=torch.empty(n,device='cuda',dtype=torch.int64)
  for b in range(2):
   for c in range(3):
    mask=(band==b)&(cls==c)
    j=torch.searchsorted(self.cdf[b,c],u[mask]).clamp_max(len(self.ix[b,c])-1)
    ix[mask]=self.ix[b,c][j]
  return ix,band,cls,(self.prior/self.proposal)[band,cls]
def loss_value(net,x,residual,band,cls,importance,mode,sampler):
 with torch.autocast('cuda',torch.bfloat16):raw=net(x)
 delta,logits,mu=decoded(raw,mode,sampler.prior,sampler.proposal)
 row=torch.arange(len(x),device='cuda')
 selected=delta[row,:,band]
 risk=((selected-residual[:,None]).square()*importance[:,None]).mean()/25
 if mode=='mse':return risk,dict(risk=risk.detach())
 selected_logits=logits[row,:,band];selected_mu=mu[row,:,band]
 classloss=F.cross_entropy(selected_logits.reshape(-1,3),cls[:,None].expand(-1,8).reshape(-1))
 amp=selected_mu.gather(-1,cls[:,None,None].expand(-1,8,1)).squeeze(-1)
 amploss=(amp-residual[:,None]).square().mean()/25
 return risk+.25*classloss+.25*amploss,dict(risk=risk.detach(),classification=classloss.detach(),amplitude=amploss.detach())
@torch.inference_mode()
def score(net,d,mode,sampler,indices=None,force_base=False):
 net.eval();s=torch.zeros(2,device='cuda',dtype=torch.float64);c=s.clone()
 groups=torch.zeros(3,2,device='cuda',dtype=torch.float64);gc=groups.clone()
 n=len(d.common) if indices is None else len(indices)
 for start in range(0,n,32768):
  ix=slice(start,start+32768) if indices is None else indices[start:start+32768]
  base=d.base[ix]
  if force_base:p=base
  else:
   with torch.autocast('cuda',torch.bfloat16):raw=net(d.x(ix))
   p=base+decoded(raw,mode,sampler.prior,sampler.proposal)[0].mean(1)
  mask=d.target[ix,2:4].bool();y=d.target[ix,:2];e=torch.where(mask,p-y,0.)
  s+=e.square().sum(0,dtype=torch.float64);c+=mask.sum(0)
  r=y-base;classes=torch.where(r>=5,1,torch.where(r<=-5,2,0))
  for k in range(3):
   mk=mask&(classes==k);groups[k]+=torch.where(mk,e.square(),0.).sum(0,dtype=torch.float64);gc[k]+=mk.sum(0)
 net.train()
 return dict(rmse=(s/c).sqrt().cpu().tolist(),sse=s.cpu().tolist(),count=c.cpu().tolist(),
  residual_group_rmse=(groups/gc.clamp_min(1)).sqrt().cpu().tolist(),residual_group_sse=groups.cpu().tolist(),residual_group_count=gc.cpu().tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['mse','state'],default='state');ap.add_argument('--check',action='store_true')
 a=ap.parse_args();gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 sm=Sampler();z=np.load(NORM);args=[torch.tensor(z['mean']),torch.tensor(z['std']),sm.prior.cpu(),sm.proposal.cpu(),sm.means.cpu()]
 if a.check:
  t=time.monotonic();d=Data('fit','terrain');net=create(a.mode,*args).cuda()
  gen=torch.Generator(device='cuda').manual_seed(104073)
  ix,b,c,w=sm.sample(262144,gen);row=torch.arange(len(ix),device='cuda');r=d.target[ix,:2]-d.base[ix]
  actual=torch.where(r[row,b]>=5,1,torch.where(r[row,b]<=-5,2,0))
  assert torch.equal(actual,c) and bool(d.target[ix,2:4][row,b].all())
  sampled=[];exact=[]
  for bb in range(2):
   mask=(b==bb);estimate=(w[mask]*r[row[mask],b[mask]].square()).mean()
   valid=d.target[:,2+bb].bool();ww=d.weight*valid;rr=d.target[:,bb]-d.base[:,bb]
   truth=(ww*rr.square()).sum()/ww.sum()
   sampled.append(float(estimate));exact.append(float(truth))
   assert abs(float(estimate/truth)-1)<.12
  xx=d.x(ix[:16384]);initial=[]
  for mode in ['mse','state']:
   model=create(mode,*args).cuda();model.eval()
   with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):p=decoded(model(xx),mode,sm.prior,sm.proposal)[0]
   initial.append(float(p.abs().max()));assert float(p.abs().max())<1e-5
   del model
  opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001,fused=True);tt=[]
  for j in range(12):
   ix,b,c,w=sm.sample(16384,gen);row=torch.arange(len(ix),device='cuda');r=(d.target[ix,:2]-d.base[ix])[row,b]
   torch.cuda.synchronize();t0=time.monotonic();opt.zero_grad(set_to_none=True)
   loss,parts=loss_value(net,d.x(ix),r,b,c,w,a.mode,sm);loss.backward();assert torch.isfinite(loss);opt.step()
   torch.cuda.synchronize();tt.append(time.monotonic()-t0)
  out=dict(GPU=gpu,mode=a.mode,sample_risk_estimate=sampled,exact_weighted_risk=exact,initial_max_abs_correction=initial,
   seconds_per_update=float(np.mean(tt[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-t)
  atomic_json(A/'check.json',out);print(json.dumps(out),flush=True);return
 plan=json.loads((A/'protocol.json').read_text());net=create(a.mode,*args);init_hash=digest(net)
 assert init_hash==plan['initial_state_sha256'][a.mode]
 out=R/'runs'/('c104_'+a.mode);out.mkdir(exist_ok=False);net=net.cuda()
 opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001,fused=True)
 t=time.monotonic();train=Data('fit','terrain');monitor=Data('monitor','terrain')
 cfg=dict(mode=a.mode,GPU=gpu,initial_state_sha256=init_hash,parameters=sum(p.numel() for p in net.parameters()),steps=plan['steps'],batch=plan['batch'],
  seed=104041,sample_seed=104073,features=235,fit_points=len(train.common),monitor_points=len(monitor.common),
  input='Same169local+66terrainPCAH/TXfeatures,C82 frozen inputs. Labels/masks not inputs.',
  method='proper3stateprior-correctedconditionalmeans' if a.mode=='state' else 'signedresidualMSE, same rare-eventimportance sampler',
  selection='minimumjointmeanmonitorRMSE,includeC82;fixedbudget;freeze beforeconfirm',official_val_used=False)
 atomic_json(out/'config.json',cfg)
 initial=score(net,monitor,a.mode,sm,force_base=True);atomic_json(out/'initial.json',initial)
 best=float(np.mean(initial['rmse']));beststep=0;history=[]
 torch.save(dict(state=net.state_dict(),step=0,config=cfg),out/'best.pt')
 gen=torch.Generator(device='cuda').manual_seed(104073);checkgen=torch.Generator(device='cuda').manual_seed(104089)
 fitix=torch.randint(len(train.common),(131072,),device='cuda',generator=checkgen)
 atomic_json(out/'fit_initial.json',score(net,train,a.mode,sm,indices=fitix,force_base=True))
 print('INITIAL',a.mode,initial['rmse'],flush=True)
 for step in range(1,plan['steps']+1):
  ix,band,cls,importance=sm.sample(plan['batch'],gen);row=torch.arange(len(ix),device='cuda')
  residual=(train.target[ix,:2]-train.base[ix])[row,band]
  lr=.0005*min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/plan['steps'])))
  for pg in opt.param_groups:pg['lr']=lr
  opt.zero_grad(set_to_none=True);loss,parts=loss_value(net,train.x(ix),residual,band,cls,importance,a.mode,sm)
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();assert torch.isfinite(loss)
  if step%250==0 or step==plan['steps']:
   val=score(net,monitor,a.mode,sm);criterion=float(np.mean(val['rmse']))
   if criterion<best:
    best=criterion;beststep=step;torch.save(dict(state=net.state_dict(),step=step,config=cfg),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   rec=dict(step=step,monitor=val,gain_vs_C82=(np.array(initial['rmse'])-val['rmse']).tolist(),best_step=beststep,
    loss=float(loss),loss_parts={k:float(v) for k,v in parts.items()},seconds=time.monotonic()-t)
   if step%500==0:rec['fit_panel']=score(net,train,a.mode,sm,indices=fitix)
   history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print(json.dumps(rec),flush=True)
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 val=score(net,monitor,a.mode,sm,force_base=beststep==0);fit=score(net,train,a.mode,sm,indices=fitix,force_base=beststep==0)
 result=dict(status='complete',best_step=beststep,last_step=plan['steps'],monitor=val,fit_panel=fit,
  gain_vs_C82=(np.array(initial['rmse'])-val['rmse']).tolist(),seconds=time.monotonic()-t,official_val_used=False)
 atomic_json(out/'result.json',result);print('DONE',a.mode,json.dumps(result),flush=True)
if __name__=='__main__':main()
