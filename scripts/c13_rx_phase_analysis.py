"""Frozen FIT predictions and train-only conditional residual calibration by RX height fraction."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
from pathlib import Path
import sys,json,csv,time
import numpy as np
import torch,lightgbm as lgb
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard
from gpu_forest import Forest
D=R/'features/c6_points';X=R/'features/c10_points';O=R/'audit/c13_rx_geometry_probe';O.mkdir(exist_ok=True)
device=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8)
run=R/'runs/c10_points_phase_geometry_s20261101';models=[Forest(lgb.Booster(model_file=str(run/f'band{b}.txt'))) for b in ['415','58']]
x={p:np.load(X/f'{p}_x.npy',mmap_mode='r') for p in ['fit','dev']};fs={p:np.load(D/f'{p}_fspl.npy',mmap_mode='r') for p in x};y={p:np.load(D/f'{p}_y.npy',mmap_mode='r') for p in x};mask={p:np.load(D/f'{p}_mask.npy',mmap_mode='r').astype(bool) for p in x}
pred={'dev':np.column_stack([np.load(run/f'dev_pred{b}.npy') for b in ['415','58']])}
test=torch.from_numpy(np.array(x['dev'][:4096])).cuda()
for k,m in enumerate(models):
 pr=m.predict(test).float().cpu().numpy()+fs['dev'][:4096,k];assert np.array_equal(pr,pred['dev'][:4096,k])
del test
p=O/'fit_c10_prediction.npy';assert not p.exists();out=np.lib.format.open_memmap(p,mode='w+',shape=y['fit'].shape,dtype=np.float32)
start=time.time()
for lo in range(0,len(out),65536):
 sl=slice(lo,min(lo+65536,len(out)));v=torch.from_numpy(np.array(x['fit'][sl])).cuda()
 for k,m in enumerate(models):out[sl,k]=m.predict(v).float().cpu().numpy()+fs['fit'][sl,k]
 del v
out.flush();pred['fit']=out
print(json.dumps({'FIT_frozen_prediction_complete':True,'seconds':time.time()-start}),flush=True)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
frac={}
for part in x:
 ids=np.load(D/f'{part}_map_indices.npy');pix=np.load(D/f'{part}_pixel.npy',mmap_mode='r').reshape(len(ids),-1)
 f=np.empty(pix.shape,np.float32)
 for j,i in enumerate(ids):
  h=H[int(rows[i]['hm_row'])];f[j]=np.remainder(h.reshape(-1)[pix[j]],1)
 frac[part]=f.reshape(-1)
meta=json.loads((X/'metadata.json').read_text());i_crit=meta['features'].index('phase0_critical_RX');i_cos=meta['features'].index('incidence_cos')
crit_edges=np.array([-2,-1,-.75,-.5,-.25,0,.25,.5,1,2,4,8]);cos_edges=np.array([-.1,-.03,0,.03,.1])
group={p:np.digitize(2*np.sinh(3*x[p][:,i_crit]),crit_edges)*(len(cos_edges)+1)+np.digitize(x[p][:,i_cos],cos_edges) for p in x}
ng=(len(crit_edges)+1)*(len(cos_edges)+1);nr=16;rng=np.random.default_rng(20261101)
rxbin={p:np.minimum((frac[p]*nr).astype(int),nr-1) for p in x}
shuffled={p:rng.permutation(rxbin[p]) for p in x}
results={}
for k,b in enumerate(['415','58']):
 m=mask['fit'][:,k];res=y['fit'][:,k].astype(float)-pred['fit'][:,k].astype(float)
 count=np.bincount(group['fit'][m],minlength=ng);sums=np.bincount(group['fit'][m],weights=res[m],minlength=ng);base=sums/(count+100)
 corrections={'geometry_only':base[group['dev']]}
 for name,bins in [('RX_fraction',rxbin),('shuffled_RX_fraction',shuffled)]:
  code=group['fit']*nr+bins['fit'];count2=np.bincount(code[m],minlength=ng*nr);sum2=np.bincount(code[m],weights=res[m],minlength=ng*nr);table=(sum2+100*np.repeat(base,nr))/(count2+100);corrections[name]=table[group['dev']*nr+bins['dev']]
 m=mask['dev'][:,k];err=pred['dev'][:,k].astype(float)-y['dev'][:,k].astype(float);oldrmse=float(np.sqrt(np.mean(err[m]**2)))
 byphase=[]
 for j in range(nr):
  keep=m&(rxbin['dev']==j);byphase.append({'fraction_bin':[j/nr,(j+1)/nr],'n':int(keep.sum()),'mean_error':float(err[keep].mean()),'RMSE':float(np.sqrt(np.mean(err[keep]**2)))})
 results[b]={'C10_DEV_sample_RMSE':oldrmse,'C10_FIT_sample_RMSE':float(np.sqrt(np.mean(res[mask['fit'][:,k]]**2))),'DEV_corrections':{n:{'RMSE':float(np.sqrt(np.mean((err[m]+c[m])**2))),'relative_RMSE_reduction_percent':float(100*(1-np.sqrt(np.mean((err[m]+c[m])**2))/oldrmse))} for n,c in corrections.items()},'DEV_error_by_RX_fraction':byphase}
result={'question':'Does RX height fractional information explain heldout C10 residual after conditioning on critical RX clearance and incidence?','scope':'Fixed FIT/DEV sampled pixels only; corrections estimated on FIT residuals, evaluated on DEV; no official VAL; weights frozen','calibration':'13 critical-clearance bins x6 incidence bins, optional16 RX-fraction bins, shrinkage100; shuffled fraction negative control','GPU':device,'GPU_peak_GB':torch.cuda.max_memory_allocated()/1e9,'no_model_weight_training':True,'seconds':time.time()-start,'bands':results}
(O/'rx_fraction_residual_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
