"""CPU diagnostics of completed C43, not an extra model or validation selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c43_architecture_20261102';proto=json.loads((A/'protocol.json').read_text())
ids=proto['dev64_ids'];G=np.load(R/'features/c3/train.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
preds={v:np.load(R/'runs'/f'c43_{v}_fit256/fp32/dev64_full.npy',mmap_mode='r') for v in ['cnn','grid','path']}
reference={}
for d in (R/'runs/c38_1024_continue/predictions').glob('train_dev_c34_fresh_shard*'):
 rows=json.loads((d/'indices.json').read_text())['index_rows'];arrays=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(rows):reference[i]=(arrays,j)
names=['cnn','grid','path','C38']
def new():
 return {'SSE':np.zeros(2),'count':np.zeros(2),'biasSSE':np.zeros(2),
         'zones':{z:[np.zeros(2),np.zeros(2)] for z in ['clear_proxy','boundary_proxy','blocked_proxy']},
         'grad_pred':np.zeros(2),'grad_true':np.zeros(2),'grad_error':np.zeros(2),'grad_cross':np.zeros(2),'grad_count':np.zeros(2),
         'band_diff_SSE':0.,'band_diff_count':0}
s={n:new() for n in names}
for j,i in enumerate(ids):
 gt=np.stack([np.asarray(y[i],float) for y in Y]);mask=np.stack([np.unpackbits(m[i],bitorder='big').reshape(256,256).astype(bool) for m in M])
 emax=(2*np.sinh(np.asarray(G[i,:8],np.float32)*3)).max(0)
 zones=[emax<-.25,abs(emax)<=.25,emax>.25]
 ar,k=reference[i];allpred={v:np.asarray(a[j]).T.reshape(2,256,256).astype(float) for v,a in preds.items()}
 allpred['C38']=np.stack([np.asarray(v[k],float) for v in ar])
 for n,pr in allpred.items():
  a=s[n];err=pr-gt
  for b in range(2):
   valid=mask[b];v=err[b][valid];a['SSE'][b]+=v@v;a['count'][b]+=len(v);a['biasSSE'][b]+=v.sum()**2/len(v)
   for z,key in zip(zones,a['zones']):
    keep=z&valid;v0=err[b][keep];a['zones'][key][0][b]+=v0@v0;a['zones'][key][1][b]+=keep.sum()
   for axis in [0,1]:
    dp=np.diff(pr[b],axis=axis);dy=np.diff(gt[b],axis=axis)
    keep=mask[b][:-1,:]&mask[b][1:,:] if axis==0 else mask[b][:,:-1]&mask[b][:,1:]
    dp=dp[keep];dy=dy[keep];de=dp-dy
    a['grad_pred'][b]+=dp@dp;a['grad_true'][b]+=dy@dy;a['grad_error'][b]+=de@de;a['grad_cross'][b]+=dp@dy;a['grad_count'][b]+=len(dp)
  common=mask[0]&mask[1];de=(pr[1]-pr[0])-(gt[1]-gt[0]);v=de[common]
  a['band_diff_SSE']+=v@v;a['band_diff_count']+=len(v)
out={'scope':'same frozen DEV64,264 full maps; C43 uses256FIT terrains, C38 usesall5508FIT terrains; diagnostics only','models':{}}
for n,a in s.items():
 out['models'][n]={
   'RMSE':np.sqrt(a['SSE']/a['count']).tolist(),
   'oracle_per_map_bias_removed_RMSE':np.sqrt(np.maximum(a['SSE']-a['biasSSE'],0)/a['count']).tolist(),
   'SSE_fraction_explained_by_oracle_map_bias':(a['biasSSE']/a['SSE']).tolist(),
   'zones':{z:{'RMSE':np.sqrt(v[0]/v[1]).tolist(),'count':v[1].astype(int).tolist()} for z,v in a['zones'].items()},
   'adjacent_difference_RMSE':np.sqrt(a['grad_error']/a['grad_count']).tolist(),
   'adjacent_difference_amplitude_ratio':np.sqrt(a['grad_pred']/a['grad_true']).tolist(),
   'adjacent_difference_correlation':(a['grad_cross']/np.sqrt(a['grad_pred']*a['grad_true'])).tolist(),
   'two_band_difference_RMSE':float(np.sqrt(a['band_diff_SSE']/a['band_diff_count']))}
out['limits']='Zones use input-only nominal3m TX raster clearance, not true propagation-state labels. Per-map bias removal uses labels as an oracle diagnostic and is not a usable correction. Gradient metrics supplement, not replace, maskedRMSE.'
(A/'error_decomposition256.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
