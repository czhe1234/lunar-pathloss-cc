"""CPU-only bounds for local remeasurement of a frozen C82 power field."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c139_bottleneck'
start=time.monotonic();result={}
for part,path,idxpath in [('fit','audit/c101_residual/base0of1/prediction.npy','audit/c101_residual/base0of1/indices.json'),('monitor','audit/c88_mesh/base.npy','audit/c88_mesh/base_index.json')]:
 d=json.loads((A/part/'meta.json').read_text());ids=d['ids'];pp=d['points_per_map'];im=json.loads((R/idxpath).read_text());im=im['ids'] if isinstance(im,dict) else im;lookup={i:j for j,i in enumerate(im)}
 fields=np.load(R/path,mmap_mode='r');targets=np.load(A/part/'target.npy',mmap_mode='r').reshape(-1,pp,4);base=np.load(A/part/'base.npy',mmap_mode='r').reshape(-1,pp,2);px=np.load(A/part/'pixels.npy',mmap_mode='r').reshape(-1,pp)
 sums={rad:dict(ss=np.zeros(2),nn=np.zeros(2),tailss=np.zeros(2),tailnn=np.zeros(2),below=np.zeros(2),above=np.zeros(2),unreachable_tail1=np.zeros(2)) for rad in [0,1,2,4,8]}
 records=[]
 for j,i in enumerate(ids):
  field=fields[lookup[i]].astype('float64');ix=px[j];old=field.reshape(2,-1)[:,ix].T
  assert np.max(abs(old-base[j]))<1e-4,(part,i,np.max(abs(old-base[j])))
  y=targets[j,:,:2].astype('float64');v=targets[j,:,2:]>0;tail=v&(abs(base[j]-y)>=5)
  for rad in sums:
   low=minimum_filter(field,size=(1,2*rad+1,2*rad+1),mode='nearest').reshape(2,-1)[:,ix].T
   high=maximum_filter(field,size=(1,2*rad+1,2*rad+1),mode='nearest').reshape(2,-1)[:,ix].T
   e=np.maximum(y-high,0)+np.minimum(y-low,0);s=sums[rad]
   s['ss']+=(e*e*v).sum(0);s['nn']+=v.sum(0);s['tailss']+=(e*e*tail).sum(0);s['tailnn']+=tail.sum(0)
   s['below']+=((y<low)&v).sum(0);s['above']+=((y>high)&v).sum(0);s['unreachable_tail1']+=((abs(e)>1)&tail).sum(0)
   if rad in [0,2,4]:records.append(dict(id=i,radius=rad,SSE=(e*e*v).sum(0).tolist(),n=v.sum(0).tolist()))
  if j%100==0:print(part,j,len(ids),'seconds',round(time.monotonic()-start,1),flush=True)
 result[part]={'radii':{rad:dict(optimistic_RMSE=np.sqrt(s['ss']/s['nn']).tolist(),tail_RMSE=np.sqrt(s['tailss']/s['tailnn']).tolist(),
  truth_stronger_than_local_range_fraction=(s['below']/s['nn']).tolist(),truth_weaker_than_local_range_fraction=(s['above']/s['nn']).tolist(),
  tail_impossible_within1db_fraction=(s['unreachable_tail1']/s['tailnn']).tolist()) for rad,s in sums.items()},'per_map':records}
out=dict(status='complete',scope='TRAIN-only label-assisted expressibility check; NOT a deployable prediction or evidence that offset is predictable',
 question='Can a nonnegative local interpolation/warp of frozen C82 reach the target power even with oracle weights?',
 definition='Clip true dB target into the min/max C82 prediction within square radius r. Nonnegative normalized interpolation in either dB or power cannot leave this range. Each band is independently relaxed, making this an optimistic lower bound.',
 inputs_modified=False,geometry_and_TX_unchanged=True,not_C95_offset_search=True,results=result,seconds=time.monotonic()-start)
(A/'local_support.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v['radii'] for k,v in result.items()},ensure_ascii=False),flush=True)
