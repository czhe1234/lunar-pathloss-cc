import json,time,hashlib
from pathlib import Path
import numpy as np
from c21_data import Data,R
A=R/'audit/c21_execution_20261101';A.mkdir(exist_ok=True)
data=Data();meta=json.loads((R/'features/c17_spatial/metadata.json').read_text())
x=np.load(R/'features/c17_spatial/fit_x.npy',mmap_mode='r')
n=len(x);mean=np.zeros(45);ss=np.zeros(45);count=0;occ=[]
for j in range(n):
 v=np.asarray(x[j,:45,::8,::8],np.float64).reshape(45,-1);mean+=v.sum(1);ss+=(v*v).sum(1);count+=v.shape[1]
 occ.append(float((x[j,39]>0).mean()))
mean/=count;std=np.sqrt(np.maximum(ss/count-mean**2,0)).clip(.05)
np.savez(A/'normalization.npz',mean=mean.astype(np.float32),std=std.astype(np.float32))
order=np.argsort(occ);chosen=order[[int((n-1)*q) for q in [.1,.35,.65,.9]]]
ids=[meta['ids']['fit'][int(j)] for j in chosen]
checks=[]
for j in chosen:
 i=meta['ids']['fit'][int(j)];t=time.time();fresh,tx,fs=data.inputs(i);ref=np.asarray(x[j,:45],np.float32);diff=np.abs(fresh-ref)
 checks.append({'id':i,'seconds':time.time()-t,'max_abs_per_channel':diff.reshape(45,-1).max(1).tolist(),'rms_per_channel':np.sqrt((diff*diff).reshape(45,-1).mean(1)).tolist()})
samples=[data.sample(i) for i in ids]
for k,name in enumerate(['x','tx','fs','y','m']):np.save(A/f'small_{name}.npy',np.stack([s[k] for s in samples]))
fitset=set(data.ids['fit'].tolist());devset=set(data.ids['dev'].tolist());assert all(i in fitset for i in ids);assert not fitset&devset
result={'small_ids':ids,'small_terrain_ids':[data.rows[i]['terrain_id'] for i in ids],'selection':'one C17 FIT terrain at input-derived occlusion fractions nearest 10%,35%,65%,90% order statistics; no labels used','input_occlusion_fraction':[occ[j] for j in chosen],'fit_maps':len(fitset),'dev_maps':len(devset),'parity':checks,'normalization':'C17 FIT512 maps first45 channels only, spatial stride8, std floor0.05','input_channels':45,'c10_prediction_inputs':False}
(A/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
