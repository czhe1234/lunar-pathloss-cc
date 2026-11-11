from pathlib import Path
import json,time,sys,hashlib,csv
import numpy as np
from catboost import CatBoostRegressor
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c106_metrics import metrics
A=R/'audit/c116_targeted';P=R/'audit/c101_residual/confirm'
fr=json.loads((A/'selection_frozen.json').read_text());t=time.monotonic()
def ld(n):return np.load(P/(n+'.npy'),mmap_mode='r')
y=ld('target')[:,:2];mask=ld('target')[:,2:4].astype(bool);base=ld('base');p=np.array(base)
for band,ch in enumerate(fr['bands']):
 if not ch['eligible']:continue
 path=R/ch['model'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ch['sha256']
 model=CatBoostRegressor();model.load_model(str(path))
 for start in range(0,len(p),131072):
  sl=slice(start,start+131072);x=np.concatenate([ld('common')[sl],ld('precise')[sl]],1).astype('float32')
  p[sl,band]+=model.predict(x,task_type='CPU',thread_count=6).astype('float32')
 print('band',band,'seconds',time.monotonic()-t,flush=True)
np.save(A/'confirm_prediction.npy',p)
scores={k:metrics(v,base,y,mask) for k,v in [('C82',base),('threshold',p)]}
meta=json.loads((P/'done.json').read_text());ids=meta['ids'];pp=meta['points_per_map']
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));terr=[rows[i]['terrain_id'] for i in ids]
ss=np.stack([np.where(mask,(v.astype('float64')-y)**2,0).reshape(len(ids),pp,2).sum(1) for v in [base,p]])
ct=mask.reshape(len(ids),pp,2).sum(1);uniq,ix=np.unique(terr,return_inverse=True)
ts=np.array([[np.bincount(ix,weights=z[:,b]) for b in range(2)] for z in ss]).transpose(0,2,1)
tc=np.array([np.bincount(ix,weights=ct[:,b]) for b in range(2)]).T
rng=np.random.default_rng(116097);draw=rng.integers(0,len(uniq),(2000,len(uniq)));den=tc[draw].sum(1)
ci=np.quantile(np.sqrt(ts[0][draw].sum(1)/den)-np.sqrt(ts[1][draw].sum(1)/den),[.025,.975],axis=0).T
gain=np.array(scores['C82']['rmse'])-scores['threshold']['rmse']
np.savez(A/'confirm_sums.npz',sse=ss,count=ct,terrain=np.array(terr),mapid=np.array(ids))
out=dict(status='complete',metrics=scores,gain=gain.tolist(),paired95ci=ci.tolist(),promote=bool((gain>=.3).all() and (ci[:,0]>0).all()),official_val_used=False,seconds=time.monotonic()-t)
(A/'confirm_summary.json').write_text(json.dumps(out,indent=2));print({k:v['rmse'] for k,v in scores.items()},gain,flush=True)
