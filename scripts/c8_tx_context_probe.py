"""Can input-only TX context predict label-derived boundary height?
The target is a diagnostic proxy, not a recovered actual TX height.
Train on prior FIT confirmation terrains; test on untouched DEV terrains.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from functools import lru_cache
import json,csv
import numpy as np,lightgbm as lgb
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
records=json.loads((R/'audit/c4_remaining_error_20261031/train_height_records.json').read_text())
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r')
dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
@lru_cache(maxsize=96)
def fields(i):
 h=np.asarray(H[i]);sy,sx=np.gradient(h);syy,syx=np.gradient(sy);sxy,sxx=np.gradient(sx)
 stats=np.r_[h.mean(),h.std(),h.max(),np.quantile(h,[.1,.25,.5,.75,.9])]
 return h,sx,sy,sxx,syy,sxy,stats
X=[];Y=[];ids=[];labels=[]
for rec in records:
 a,b=rec['bands']
 if a is None or b is None:continue
 height=(a['aopt']+b['aopt'])/2
 if min(a['clear_n'],b['clear_n'],a['blocked_n'],b['blocked_n'])<30:continue
 if max(a['near_optimal_width'],b['near_optimal_width'])>.5 or abs(a['aopt']-b['aopt'])>.25:continue
 i=rec['index_row'];r=rows[i];tr=int(r['tx_row']);tc=int(r['tx_col']);h,sx,sy,sxx,syy,sxy,stats=fields(int(r['hm_row']))
 vals=[h[tr,tc],tr/256,tc/256,sx[tr,tc],sy[tr,tc],sxx[tr,tc],syy[tr,tc],sxy[tr,tc],*stats]
 for dr in [-4,-2,-1,0,1,2,4]:
  for dc in [-4,-2,-1,0,1,2,4]:vals.append(h[np.clip(tr+dr,0,255),np.clip(tc+dc,0,255)]-h[tr,tc])
 for q in [.25,.5,1,2]:
  vals.extend([np.sin(2*np.pi*h[tr,tc]/q),np.cos(2*np.pi*h[tr,tc]/q)])
 X.append(vals);Y.append(height);ids.append(r['terrain_id']);labels.append(i)
X=np.array(X,np.float32);Y=np.array(Y);ids=np.array(ids);fit=np.array([i not in dev for i in ids]);check=~fit
assert set(ids[fit]).isdisjoint(set(ids[check]))
u=np.unique(ids[fit]);rng=np.random.default_rng(20261101);rng.shuffle(u);inner=np.isin(ids,u[:max(1,len(u)//4)])&fit
params=dict(objective='regression',metric='rmse',num_leaves=15,max_depth=5,min_data_in_leaf=25,learning_rate=.03,lambda_l2=5,verbosity=-1,num_threads=8,device_type='cpu',seed=20261101,deterministic=True,force_col_wise=True)
tmp=lgb.train(params,lgb.Dataset(X[fit&~inner],label=Y[fit&~inner]),num_boost_round=500,valid_sets=[lgb.Dataset(X[inner],label=Y[inner])],callbacks=[lgb.early_stopping(50,verbose=False)])
model=lgb.train(params,lgb.Dataset(X[fit],label=Y[fit]),num_boost_round=tmp.best_iteration)
pred=model.predict(X);mean=Y[fit].mean();score=lambda v:{'n':int(v.sum()),'terrains':len(np.unique(ids[v])),'proxy_height_std_m':float(Y[v].std()),'constant_RMSE_m':float(np.mean((Y[v]-mean)**2)**.5),'model_RMSE_m':float(np.mean((Y[v]-pred[v])**2)**.5),'R2':float(1-np.mean((Y[v]-pred[v])**2)/np.var(Y[v]))}
o=R/'audit/c8_tx_context_probe';o.mkdir(exist_ok=True);model.save_model(str(o/'model.txt'))
res={'target':'Mean of two-band label-derived optimal geometric boundary heights. Not actual TX height.','filter':'both clear and blocked >=30, both near-optimal widths <=0.5m, band height agreement <=0.25m','feature_count':X.shape[1],'best_iteration_inner_FIT':tmp.best_iteration,'constant_height_m':mean,'fit':score(fit),'dev':score(check),'dev_used_for_fitting_or_stopping':False}
(o/'result.json').write_text(json.dumps(res,indent=2)+'\n');np.savez(o/'predictions.npz',index_rows=labels,proxy=Y,pred=pred,is_fit=fit);print(json.dumps(res,indent=2),flush=True)

