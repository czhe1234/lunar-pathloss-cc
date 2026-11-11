"""CPU-only attribution of inverse gains and source-gradient conflicts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time,math
import numpy as np,torch
torch.set_num_threads(2)
R=Path(__file__).resolve().parents[1]
A=R/'audit/c139_bottleneck'
def save(p,d):
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def scatter(ix,w,n):
 return np.stack([np.bincount(ix,weights=w[:,b],minlength=n) for b in range(2)],1)
def main():
 t0=time.monotonic();ck=torch.load(R/'runs/c137_oracle16.pt',map_location='cpu',weights_only=False);ids=ck['ids']
 P=R/'audit/c101_residual/fit';meta=json.loads((P/'done.json').read_text());pp=meta['points_per_map'];idx={v:i for i,v in enumerate(meta['ids'])}
 tt=np.load(P/'target.npy',mmap_mode='r').reshape(-1,pp,4);bb=np.load(P/'base.npy',mmap_mode='r').reshape(-1,pp,2)
 ss=np.zeros((8,2));nn=np.zeros(2);tss=np.zeros((8,2));tn=np.zeros(2);shares=np.zeros((3,2));share_tail=np.zeros((3,2));conflict=[];records=[]
 for j,i in enumerate(ids):
  with np.load(R/'audit/c136_native/cache'/f'{i}.npz') as z:
   g={k:z[k] for k in ['si','sw','ai','aw','dw','uf','inv','pw','active','feat']}
  y=tt[idx[i],:,:2].astype('float64');v=tt[idx[i],:,2:]>0;base=bb[idx[i]].astype('float64');res=y-base;tail=v&(abs(res)>=5)
  nr=len(g['uf']);si=g['si'];ai=g['ai'];N=len(g['feat'])
  logs=(8*torch.tanh(ck['states'][f'{j}.raw']/8)).numpy().astype('float64');dl=(8*torch.tanh(ck['states'][f'{j}.draw']/8)).numpy().astype('float64')
  neutral=[g['dw'].astype('float64'),scatter(si[:,0],g['sw'],nr),scatter(ai[:,0],g['aw'],nr)]
  changed=[neutral[0]*np.exp(dl),scatter(si[:,0],g['sw']*np.exp(logs[si[:,1],:2]),nr),scatter(ai[:,0],g['aw']*np.exp(logs[:,2:].reshape(-1,2)[ai[:,1]]),nr)]
  def pix(z):return (z[g['inv']]*g['pw']).sum(1)
  part=[pix(z) for z in neutral]
  for k in range(3):
   shares[k]+=(part[k]*v).sum(0);share_tail[k]+=(part[k]*tail).sum(0)
  row=[]
  for bits in range(8):
   z=sum(changed[k] if bits&(1<<k) else neutral[k] for k in range(3))
   ratio=np.where(g['active'],pix(z),1)
   pred=base-10/np.log(10)*np.log(np.maximum(ratio,1e-12))
   e=np.where(v,pred-y,0);s=(e*e).sum(0);ss[bits]+=s;tss[bits]+=(e*e*tail).sum(0);row.append(s.tolist())
  nn+=v.sum(0);tn+=tail.sum(0)
  # Linear gradient at neutral. A source gets a sum of opposing RX residual demands.
  signed=scatter(g['inv'].reshape(-1),(g['pw']*np.where(v,res,0)[:,None]).reshape(-1,2),nr)
  unsigned=scatter(g['inv'].reshape(-1),(g['pw']*np.where(v,abs(res),0)[:,None]).reshape(-1,2),nr)
  # Existing per-source scalar vs a diagnostic split by geometric source->RX direction.
  # This direction is an endpoint direction only, not actual mixed-path departure angle.
  for name,ix,wt,m,div in [('spec',si,g['sw'],N,1),('diffuse',ai,g['aw'],N*2,2)]:
   src=ix[:,1];rx=ix[:,0]
   p=scatter(src,wt*signed[rx],m);ab=scatter(src,wt*unsigned[rx],m)
   coherent=np.abs(p).sum(0)/np.maximum(ab.sum(0),1e-30)
   delta=g['feat'][g['uf'][rx],:3].astype('float32')-g['feat'][src//div,:3].astype('float32')
   angle=np.arctan2(delta[:,1],delta[:,0]);binning=np.minimum(7,((angle+np.pi)*8/(2*np.pi)).astype('int32'))
   flat=(src.astype('int64')*8+binning);uniq,inv=np.unique(flat,return_inverse=True)
   pa=scatter(inv,wt*signed[rx],len(uniq));aba=scatter(inv,wt*unsigned[rx],len(uniq))
   angular=np.abs(pa).sum(0)/np.maximum(aba.sum(0),1e-30)
   conflict.append(dict(id=i,component=name,source_non_cancellation=coherent.tolist(),endpoint_8bin_non_cancellation=angular.tolist(),denominator=ab.sum(0).tolist()))
  records.append(dict(id=i,SSE=row,valid=v.sum(0).tolist()))
  print('DONE_MAP',i,j+1,len(ids),'seconds',round(time.monotonic()-t0,2),flush=True)
 names=['neutral','direct','specular','direct+specular','diffuse','direct+diffuse','specular+diffuse','all']
 rm=np.sqrt(ss/nn);shapley=np.zeros((3,2))
 for k in range(3):
  for bits in range(8):
   if bits&(1<<k):continue
   s=bits.bit_count();weight=math.factorial(s)*math.factorial(2-s)/6
   shapley[k]+=weight*(ss[bits]-ss[bits|(1<<k)])
 out=dict(status='complete',scope='16 FIT maps,1536 fixed points/map; label-assisted attribution, NOT heldout score or deployable model',ids=ids,valid=nn.tolist(),baseline=rm[0].tolist(),
  variants={n:dict(RMSE=rm[k].tolist(),tail_RMSE=np.sqrt(tss[k]/tn).tolist()) for k,n in enumerate(names)},
  neutral_power_fraction=(shares/nn).tolist(),neutral_tail_power_fraction=(share_tail/tn).tolist(),
  SSE_gain_shapley_fraction=(shapley/np.maximum(ss[0]-ss[7],1e-30)).tolist(),components=['direct','specular','diffuse'],
  conflicts=conflict,per_map=records,seconds=time.monotonic()-t0,
  limits=['Direction split is label-assisted gradient diagnostic; increased freedom necessarily reduces cancellation and is not proof of generalization.','Endpoint direction differs from actual departure for mixed paths.','Original C137 allows free direct receiver gains and free per-map source states; neither establishes predictability from H/TX.'])
 save(A/'attribution.json',out)
 print(json.dumps({k:v for k,v in out.items() if k not in ['conflicts','per_map']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
