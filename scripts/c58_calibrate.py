"""FIT-only calibration of straight-transmission + high-band-scattering proxies."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parents[1];O=R/'features/c58_transmission';A=R/'audit/c58_exact_transmission_20261102'
data={}
for part in ['fit','dev']:
 meta=json.loads((O/(part+'_meta.json')).read_text());assert meta['complete'] and meta['input_only'] and not meta['PL_or_validity_masks_read']
 ids=np.load(O/(part+'_point_indices.npy'));folder='c38_dense' if part=='fit' else 'c6_points'
 yy=np.load(R/f'features/{folder}/{part}_y.npy',mmap_mode='r')[ids]
 fs=np.load(R/f'features/{folder}/{part}_fspl.npy',mmap_mode='r')[ids]
 mask=np.load(R/f'features/{folder}/{part}_mask.npy',mmap_mode='r')[ids].astype(bool)
 base=np.load(R/f'features/c47_teacher/{part}_base.npy',mmap_mode='r')[ids].astype(float)*20
 data[part]={'y':yy.astype(float)-fs.astype(float),'base':base,'mask':mask,
  'loss':np.load(O/(part+'_loss.npy')).reshape(len(ids),4,2).astype(float),
  'counts':np.load(O/(part+'_counts.npy')).reshape(len(ids),4)}
def score(d,pred):
 m=d['mask'][:,0];e=pred[m]-d['y'][m,0];return float(np.sqrt(np.mean(e*e)))
def physics(d,variant,offset):
 tl=d['loss'][:,variant,0]
 p=-10/np.log(10)*np.logaddexp(-np.log(10)*tl/10,-np.log(10)*(d['base'][:,1]+offset)/10)
 return np.where(d['counts'][:,variant]>0,p,d['base'][:,0])
f=data['fit'];m=f['mask'][:,0];res=f['y'][m,0]-f['base'][m,0];best=None;trials=[]
for variant in range(4):
 for offset in [-12,-8,-4,0,4,8,12]:
  p=physics(f,variant,offset);delta=p[m]-f['base'][m,0]
  weight=float(np.clip(delta@res/max(delta@delta,1e-12),0,1));pred=f['base'][:,0]+weight*(p-f['base'][:,0]);fit=score(f,pred)
  trial={'variant':variant,'height':'nominal3' if variant<2 else 'raster_phase_candidate',
    'normal':['diag_main','diag_anti'][variant%2],'scattering_offset_db':offset,'weight':weight,'FIT_RMSE':fit}
  trials.append(trial)
  if best is None or fit<best['FIT_RMSE']:best=trial
records={}
for part,d in data.items():
 p=physics(d,best['variant'],best['scattering_offset_db'])
 pred=d['base'][:,0]+best['weight']*(p-d['base'][:,0])
 mask=d['mask'][:,0];blocked=d['counts'][:,best['variant']]>0
 records[part]={'C38_RMSE':score(d,d['base'][:,0]),'calibrated_RMSE':score(d,pred),'gain_db':score(d,d['base'][:,0])-score(d,pred),
 'valid_points':int(mask.sum()),'geometric_crossing_points':int((mask&blocked).sum()),
 'raw_transmission_RMSE_on_crossing_points':float(np.sqrt(np.mean((d['loss'][mask&blocked,best['variant'],0]-d['y'][mask&blocked,0])**2)))}
# Another FIT-only low-dimensional test; no DEV coefficient tuning.
D={part:np.stack([physics(d,v,0)-d['base'][:,0] for v in range(4)],1) for part,d in data.items()}
X=D['fit'][m];gram=X.T@X;beta=np.linalg.solve(gram+np.eye(4)*.01*np.trace(gram)/4,X.T@res);beta=np.clip(beta,-.5,.5)
ridge={}
for part,d in data.items():
 pr=d['base'][:,0]+D[part]@beta
 ridge[part]={'RMSE':score(d,pr),'gain_db':score(d,d['base'][:,0])-score(d,pr)}
out={'scope':'FIT64 calibration and terrain-disjointDEV64 sampled points;415MHz only. Not officialVAL',
 'selected_on_FIT_only':best,'records':records,'ridge_beta':beta.tolist(),'ridge':ridge,'promotion_low_gain010db':bool(max(records['dev']['gain_db'],ridge['dev']['gain_db'])>=.1),
 'limits':'Exact candidate-mesh events; two facet choices. No original mesh recovered. High-band predictor is frozen C38; no high-band target used as input.'}
(A/'calibration.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
