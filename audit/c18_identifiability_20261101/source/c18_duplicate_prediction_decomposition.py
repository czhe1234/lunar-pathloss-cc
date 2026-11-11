"""Frozen C10 inference on exact input repeats, followed by algebraic error decomposition."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
from pathlib import Path
import sys,csv,json,time,hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from c10_inference import C10Predictor
from cc_core import check_source_unchanged
D=R.parent/'data';O=R/'audit/c18_identifiability_20261101';records=json.loads((O/'identical_input_audit.json').read_text())['records']
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r')
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
model=C10Predictor(R/'runs/c10_points_phase_geometry_s20261101')
class Tap:
 def __init__(self,f):self.f=f;self.last=None
 def predict(self,x):
  self.last=x[:,11:20].cpu().numpy();return self.f.predict(x)
tap=Tap(model.forests[0]);model.forests[0]=tap
out=[];pred=[];start=time.time();check_source_unchanged()
for rec in records:
 ii=rec['index_rows'];assert len(ii)==2
 row=rows[ii[0]];h=H[int(row['hm_row'])].astype(np.float32);tr=int(row['tx_row']);tc=int(row['tx_col'])
 assert np.array_equal(h,H[int(rows[ii[1]]['hm_row'])])
 pr=model.predict(h,tr,tc);pred.append(pr)
 e3=(2*np.sinh(3*tap.last[:,:8])).max(1).reshape(256,256);e2=(2*np.sinh(3*tap.last[:,8])).reshape(256,256)
 zones={'all':np.ones((256,256),bool),'occluded_proxy':e2>=0,'height_sensitive':(e3<0)&(e2>=0),'nominal_blocked':e3>=0,'stable_clear':e2<-.5}
 r={'index_rows':ii,'terrain_id':row['terrain_id'],'base_weight_fit_member':row['terrain_id'] not in dev,'zones':{}}
 for zn,z in zones.items():
  r['zones'][zn]={}
  for k,b in enumerate(['415','58']):
   ys=np.array(Y[k][ii],float);ms=np.unpackbits(M[k][ii],axis=1,bitorder='big').reshape(2,256,256).astype(bool);v=ms.all(0)&z
   mean=ys.mean(0);ambiguity=((ys[0]-ys[1])**2/2)[v].sum();common=(2*(pr[k]-mean)**2)[v].sum();model_sse=(((ys-pr[k])**2).sum(0))[v].sum();n=2*int(v.sum())
   assert abs(ambiguity+common-model_sse)<1e-6*max(model_sse,1)
   r['zones'][zn][b]={'observations':n,'empirical_inconsistency_SSE':float(ambiguity),'common_model_error_SSE':float(common),'model_SSE':float(model_sse)}
 out.append(r)
pool={}
for subset in ['all','base_fit','base_unseen']:
 pool[subset]={}
 rr=[r for r in out if subset=='all' or r['base_weight_fit_member']==(subset=='base_fit')]
 for zn in zones:
  pool[subset][zn]={}
  for b in ['415','58']:
   vals=[r['zones'][zn][b] for r in rr];total={k:sum(v[k] for v in vals) for k in ['observations','empirical_inconsistency_SSE','common_model_error_SSE','model_SSE']}
   n=total['observations']
   total.update({'model_RMSE':float(np.sqrt(total['model_SSE']/n)) if n else None,'empirical_inconsistency_RMSE':float(np.sqrt(total['empirical_inconsistency_SSE']/n)) if n else None,'empirical_inconsistency_fraction_of_model_SSE':total['empirical_inconsistency_SSE']/max(total['model_SSE'],1e-15)})
   pool[subset][zn][b]=total
np.save(O/'duplicate_C10_predictions.npy',np.stack(pred))
result={'state':'completed','scope':'9 exact H+TX duplicate pairs, joint-valid pixels; FIT-only C10 frozen','records':out,'pooled':pool,
 'equation':'sum_j(pred-y_j)^2 = n*(pred-mean(y))^2 + sum_j(y_j-mean(y))^2',
 'duplicate_terrain_counts':{'base_fit':sum(r['base_weight_fit_member'] for r in out),'base_unseen':sum(not r['base_weight_fit_member'] for r in out)},
 'limits':'Not independent simulation repeats; seed schedule may be shared. This empirical inconsistency is not a population noise estimate or a bound for other terrains. Base-fitting membership is reported in duplicate_terrain_counts.',
 'GPU':model.device,'GPU_peak_GB':torch.cuda.max_memory_allocated()/1e9,'seconds':time.time()-start}
(O/'duplicate_prediction_decomposition.json').write_text(json.dumps(result,indent=2)+'\n');check_source_unchanged()
print(json.dumps({'pooled':pool,'seconds':result['seconds'],'GPU_peak_GB':result['GPU_peak_GB']}),flush=True)
