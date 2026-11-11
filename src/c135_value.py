"""Fixed-support physical calibrations and diagnostic ceilings, no official VAL."""
import sys,csv,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
import numpy as np,torch
A=R/'audit/c135_paths'
def dataset(part):
 plan=json.loads((A/'protocol.json').read_text());ids=plan['fit_probe' if part=='fit' else 'monitor']
 src=R/'audit/c101_residual'/part;m=json.loads((src/'done.json').read_text());pp=m['points_per_map'];lookup={i:j for j,i in enumerate(m['ids'])}
 y=np.load(src/'target.npy',mmap_mode='r').reshape(-1,pp,4);b=np.load(src/'base.npy',mmap_mode='r').reshape(-1,pp,2);w=np.load(src/'weight.npy',mmap_mode='r').reshape(-1,pp)
 powers=[];ys=[];bs=[];ws=[];geo=[]
 for i in ids:
  ar=np.load(A/'cache'/f'{i}.npz');powers.append(ar['power']);geo.append(ar['direct_geometry']);j=lookup[i];ys.append(y[j]);bs.append(b[j]);ws.append(w[j])
 cv=lambda x:torch.tensor(np.concatenate(x),device='cuda',dtype=torch.float32)
 p=cv(powers);target=cv(ys);base=cv(bs);weight=cv(ws).flatten()
 if part=='monitor':weight=torch.ones_like(weight)
 return dict(power=p,y=target[:,:2],v=target[:,2:]>0,base=base,w=weight,geo=cv(geo),ids=ids,pp=pp)
def stats(p,d):
 v=d['v'];e=torch.where(v,p-d['y'],0.).double();r=torch.where(v,d['base']-d['y'],0.).double();n=v.sum(0)
 tail=v&(r.abs()>=5);c=p-d['base'];need=d['y']-d['base']
 def rm(x,m):return ((x.square()*m).sum(0)/m.sum(0).clamp_min(1)).sqrt().tolist()
 return dict(RMSE=rm(e,v),C82=rm(r,v),gain=(torch.tensor(rm(r,v))-torch.tensor(rm(e,v))).tolist(),
  count=n.tolist(),tail_RMSE=rm(e,tail),tail_count=tail.sum(0).tolist(),
  tail_sign_accuracy=(((c*need)>0)&tail).sum(0).div(tail.sum(0)).tolist(),
  tail_mean_abs_correction=(c.abs()*tail).sum(0).div(tail.sum(0)).tolist(),
  clear_RMSE=rm(e,v&(d['geo'][:,:1]>=.999)),
  nonclear_RMSE=rm(e,v&(d['geo'][:,:1]<.999)))
class Calibration(torch.nn.Module):
 def __init__(self,idx):
  super().__init__();self.idx=idx;self.raw=torch.nn.Parameter(torch.zeros(len(idx),2))
 def forward(self,p,base):
  z=p[:,self.idx];frac=z/z.sum(1,keepdim=True).clamp_min(1e-30)
  factor=torch.exp(4*torch.tanh(self.raw/4))
  ratio=(frac*factor).sum(1).clamp_min(1e-20)
  delta=-10*torch.log10(ratio)
  return base+torch.where(z.sum(1)>1e-30,delta,torch.zeros_like(delta))
def raw_diagnostics(d):
 p=d['power'];old=p[:,[0,1,7]].sum(1);new=p.sum(1);added=(new-old).clamp_min(0)
 delta=-10*torch.log10((new+1e-20)/(old+1e-20));e=d['base']-d['y'];v=d['v']
 # Label-assisted only: optimal pixelwise inclusion of missing paths between zero and unit weight.
 oracle_delta=torch.minimum(torch.zeros_like(e),torch.maximum(-e,delta))
 # This is neither a learned result nor an information-theoretic lower bound.
 return dict(raw_addition=stats(d['base']+delta,d),oracle_unit_path_addition=stats(d['base']+oracle_delta,d),
  addition_absolute_db_quantiles=[torch.quantile((-delta[:,f])[v[:,f]],torch.tensor([.5,.9,.99,.999],device='cuda')).tolist() for f in range(2)],
  correction_vs_needed_cosine=[float((delta[:,f]*(-e[:,f])*v[:,f]).sum()/((delta[:,f].square()*v[:,f]).sum()*(e[:,f].square()*v[:,f]).sum()).sqrt().clamp_min(1e-20)) for f in range(2)],
  added_power_fraction=(added.sum(0)/new.sum(0)).tolist(),
  ideal_unit_addition_scope='Label-assisted per-pixel 0..1 weighting of computed added paths only. Not a learnable score or a bound on the full proposed architecture.')
def main():
 gpu=gpu_guard();torch.set_num_threads(3);start=time.monotonic();fit=dataset('fit');mon=dataset('monitor')
 result=dict(GPU=gpu,fit_maps=len(fit['ids']),monitor_maps=len(mon['ids']),monitor_terrains=128,
  official_VAL=False,metric_scope='same 628 monitor maps, existing 2048 uniform points/map; full-map gate not yet run',
  diagnostics=raw_diagnostics(mon),arms={})
 for mode,idx in [('old_paths',[0,1,7]),('mixed_paths',list(range(11)))]:
  torch.manual_seed(135041);net=Calibration(idx).cuda();opt=torch.optim.Adam(net.parameters(),lr=.03)
  gen=torch.Generator(device='cuda').manual_seed(135043);den=(fit['v']*fit['w'][:,None]).sum(0)/len(fit['y']);hist=[]
  initial=float((net(fit['power'],fit['base'])-fit['base']).abs().max());assert initial<2e-5,initial
  for step in range(1500):
   ii=torch.randint(len(fit['y']),(32768,),generator=gen,device='cuda');pred=net(fit['power'][ii],fit['base'][ii])
   loss=(((pred-fit['y'][ii]).square()*fit['v'][ii]*fit['w'][ii,None]).mean(0)/den).mean()
   opt.zero_grad();loss.backward();opt.step()
   if step%300==0:hist.append(float(loss.detach()))
  lb=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=120,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
  def closure():
   lb.zero_grad();pred=net(fit['power'],fit['base']);loss=(((pred-fit['y']).square()*fit['v']*fit['w'][:,None]).mean(0)/den).mean();loss.backward();return loss
  loss_before_lb=float(closure().detach());lb.step(closure);loss_after_lb=float(closure().detach())
  with torch.no_grad():
   pp=net(mon['power'],mon['base']);pf=net(fit['power'],fit['base']);item=dict(monitor=stats(pp,mon),fit=stats(pf,fit),factors=torch.exp(4*torch.tanh(net.raw/4)).tolist(),loss_history=hist,initial_max_C82_difference=initial,full_weighted_loss_before_after_LBFGS=[loss_before_lb,loss_after_lb])
  result['arms'][mode]=item
  out=R/'runs'/f'c135_{mode}';out.mkdir(exist_ok=True);torch.save(dict(state=net.state_dict(),idx=idx,steps=1500,seed=135041),out/'model.pt')
  np.save(out/'monitor_prediction.npy',pp.cpu().numpy().reshape(len(mon['ids']),mon['pp'],2))
  print(mode,item,flush=True)
 result['seconds']=time.monotonic()-start;atomic_json(A/'value_probe.json',result);print('DONE',result['seconds'],flush=True)
if __name__=='__main__':main()
