import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c122_power'
models=['c121_old','c121_mean','c121_donor','c122_clip128','c122_wide128','c122_wide8192']
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));P=R/'audit/c101_residual/monitor'
mapid=np.load(P/'mapid.npy',mmap_mode='r');target=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r')
out={'models':{},'comparisons':{},'scope':'true abs(Y-C82)>=5 dB, fixed128terrain TRAIN monitor; not full-population sign accuracy or RMSE','official_validation_used':False,'expanded':False}
for model in models:out['models'][model]=[json.load(open(R/f'runs/{model}/result{b}.json')) for b in range(2)]
for comparator in ['c121_mean','c121_donor','c122_wide128','c122_wide8192']:
 reference='c121_old' if comparator.startswith('c121') else 'c122_clip128'
 out['comparisons'][comparator]={'reference':reference,'bands':[]}
 for b in range(2):
  ids=np.load(R/f'runs/{reference}/monitor_rows{b}.npy');assert np.array_equal(ids,np.load(R/f'runs/{comparator}/monitor_rows{b}.npy'))
  pr=np.load(R/f'runs/{reference}/monitor_prob{b}.npy').astype('float64');pc=np.load(R/f'runs/{comparator}/monitor_prob{b}.npy').astype('float64');y=target[ids,b]-base[ids,b]>=5
  ts=np.array([rows[int(i)]['terrain_id'] for i in mapid[ids]]);terrain,ti=np.unique(ts,return_inverse=True);n=np.bincount(ti)
  correct=(pc>=.5)==y;prior=(pr>=.5)==y;delta=np.bincount(ti,weights=correct.astype(float)-prior.astype(float))
  loss=lambda p:-(y*np.log(p.clip(1e-9,1))+(1-y)*np.log((1-p).clip(1e-9,1)))
  gain=np.bincount(ti,weights=loss(pr)-loss(pc))
  draws=np.random.default_rng(122107).integers(0,len(n),(2000,len(n)));boot=delta[draws].sum(1)/n[draws].sum(1);lb=gain[draws].sum(1)/n[draws].sum(1)
  out['comparisons'][comparator]['bands'].append(dict(band=b,rows=len(ids),terrains=len(terrain),accuracy_gain=float(delta.sum()/n.sum()),accuracy_gain_95ci=np.quantile(boot,[.025,.975]).tolist(),logloss_drop=float(gain.sum()/n.sum()),logloss_drop_95ci=np.quantile(lb,[.025,.975]).tolist()))
out['gates']={}
for m in ['c122_wide128','c122_wide8192']:
 out['gates'][m]=bool(all(out['models'][m][b]['conditional_sign_accuracy']>=.8 and out['comparisons'][m]['bands'][b]['accuracy_gain']>=.08 and out['comparisons'][m]['bands'][b]['logloss_drop']>=.05 for b in range(2)))
(A/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
