"""Optimistic representation bounds for which power component must be changed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c139_bottleneck'
result={}
for part in ['fit','monitor']:
 p=A/part;meta=json.loads((p/'meta.json').read_text());pp=meta['points_per_map']
 donor=np.load(p/'donor.npy',mmap_mode='r');w=np.load(p/'weights.npy',mmap_mode='r');tt=np.load(p/'target.npy',mmap_mode='r');b=np.load(p/'base.npy',mmap_mode='r')
 sums={k:np.zeros(2) for k in ['baseSSE','nonDfloor','Dfloor','n','tailn','tailSSE','tail_unreachable1','positiveTail','positiveTail_unreachable1','powerD']}
 by=[]
 for j,i in enumerate(meta['ids']):
  sl=slice(j*pp,(j+1)*pp)
  # descriptor13/14 = log10(G0_donor/G0_pixel); 15/16 = D_donor/G0_donor.
  z=donor[sl].astype('float64');ww=w[sl].astype('float64')
  rel=10**z[:,:,13:15];fd=(ww[:,:,None]*rel*z[:,:,15:17]).sum(1).clip(0,1)
  y=tt[sl,:2].astype('float64');v=tt[sl,2:]>0;base=b[sl].astype('float64');need=y-base
  e_nonD=np.maximum(0,need+10*np.log10(np.maximum(fd,1e-30)))
  e_D=np.maximum(0,need+10*np.log10(np.maximum(1-fd,1e-30)))
  tail=v&(abs(need)>=5);pos=v&(need>=5)
  se=(need*need*v).sum(0);sn=(e_nonD*e_nonD*v).sum(0);sd=(e_D*e_D*v).sum(0);n=v.sum(0)
  for k,val in dict(baseSSE=se,nonDfloor=sn,Dfloor=sd,n=n,tailn=tail.sum(0),tailSSE=(need*need*tail).sum(0),tail_unreachable1=(tail&(e_nonD>1)).sum(0),positiveTail=pos.sum(0),positiveTail_unreachable1=(pos&(e_nonD>1)).sum(0),powerD=(fd*v).sum(0)).items():sums[k]+=val
  by.append(dict(id=i,baseSSE=se.tolist(),nonDfloor=sn.tolist(),Dfloor=sd.tolist(),n=n.tolist()))
 result[part]=dict(valid=sums['n'].tolist(),C82=np.sqrt(sums['baseSSE']/sums['n']).tolist(),
  fixed_direct_unlimited_other_power_optimistic_RMSE=np.sqrt(sums['nonDfloor']/sums['n']).tolist(),
  fixed_other_unlimited_direct_power_optimistic_RMSE=np.sqrt(sums['Dfloor']/sums['n']).tolist(),
  lower_bound_SSE_fraction_of_C82=(sums['nonDfloor']/sums['baseSSE']).tolist(),
  tail_fraction=(sums['tailn']/sums['n']).tolist(),tail_SSE_fraction=(sums['tailSSE']/sums['baseSSE']).tolist(),
  positive_tail_count=sums['positiveTail'].tolist(),positive_tail_requiring_D_change_for_1db=(sums['positiveTail_unreachable1']/sums['positiveTail']).tolist(),
  mean_direct_fraction=(sums['powerD']/sums['n']).tolist(),per_map=by)
out=dict(status='complete',formula='ymax = C82 -10log10(fixed_component_fraction). error_floor=max(0,truth-ymax). The other power component is relaxed to arbitrary nonnegative pixelwise power.',
 scope='Optimistic lower bound on specified anchored positive-power model classes only; not on H/TX information or all architectures.',
 important='D is a modeled carrier. Its free scaling may compensate any discrepancy. This does NOT prove the physical direct component is the original error source.',
 results=result)
(A/'component_bound.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='per_map'} for k,v in result.items()},ensure_ascii=False))
