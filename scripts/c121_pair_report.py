import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from catboost import CatBoostClassifier
A=R/'audit/c121_pairs';P=A/'monitor';rows=json.load(open(P/'pairs.json'));f=np.load(P/'pair_features.npz');x=f['old'];v=f['donors'];meta=f['metadata'];w=meta[:,:,:3]
mean=(v*w[:,:,:,None]).sum(2);sd=np.sqrt(((v-mean[:,:,None])**2*w[:,:,:,None]).sum(2))
oldnorm=np.load(R/'audit/c101_residual/normalization.npz')['std']
good=np.array([r['same_pixel']>=0 for r in rows]);bands=np.array([r['band'] for r in rows])
groups={'raw_H_7x7':np.arange(49),'compact_physics':np.arange(49,94),'C82_power':np.arange(94,96),'ray_geometry':np.arange(96,102),'C82_spatial_difference':np.arange(102,110),'absolute_geometry':np.arange(110,121),'precise_grazing':np.arange(121,169)}
result={'pairs':len(rows),'matched_controls':int(good.sum()),'bands':[]}
for b in range(2):
 keep=(bands==b);control=keep&good
 m=CatBoostClassifier();m.load_model(str(R/f'runs/c120_old/band{b}.cbm'))
 prob=m.predict_proba(x[keep,:2].reshape(-1,169),thread_count=4)[:,1].reshape(-1,2)
 bc={'band':b,'n':int(keep.sum()),'control_n':int(control.sum()),'old_tail_classifier':{
 'both_signs_correct':float(((prob[:,0]>=.5)&(prob[:,1]<.5)).mean()),'anchor_probability_higher':float((prob[:,0]>prob[:,1]).mean()),
 'anchor_probability_mean':float(prob[:,0].mean()),'opposite_probability_mean':float(prob[:,1].mean())},
 'geometry_strata':{str(t):int(sum(r['old_geometry_rms_z']<t for r in np.array(rows,dtype=object)[keep])) for t in [.1,.2,.4]},
 'feature_groups':{}}
 for name,cols in groups.items():
  dx=(x[:,:,cols]-x[:,0:1,cols])/oldnorm[cols]
  d=np.sqrt(np.mean(dx**2,2));a=d[control,1];c=d[control,2]
  bc['feature_groups'][name]=dict(opposite_mean=float(a.mean()),same_mean=float(c.mean()),ratio=float(a.mean()/max(1e-9,c.mean())),opposite_larger_fraction=float((a>c).mean()))
 stats={}
 for name,z in [('donor_mean',mean),('donor_spread',sd)]:
  da=abs(z[control,1]-z[control,0]);db=abs(z[control,2]-z[control,0]);rat=da.mean(0)/(db.mean(0)+1e-7)
  stats[name]=[dict(channel=int(k),ratio=float(rat[k]),opposite_mean=float(da[:,k].mean()),same_mean=float(db[:,k].mean())) for k in np.argsort(-rat)[:12]]
 bc['donor_features']=stats;result['bands'].append(bc)
picked=[];used=set()
for k in np.argsort([r['old_geometry_rms_z'] for r in rows]):
 r=rows[int(k)]
 if not good[k] or r['terrain_id'] in used:continue
 picked.append(dict(row_index=int(k),**r));used.add(r['terrain_id'])
 if len(picked)>=12:break
result['examples']=picked
(A/'pair_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='examples'},ensure_ascii=False,indent=2))
print('FIRST_EXAMPLES',json.dumps(picked[:3],ensure_ascii=False))
