"""Exploratory geometry-conditioned cross-band observation control; closed-form OLS."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/c18_identifiability_20261101';D=R/'features/c6_points'
ids=np.load(D/'dev_map_indices.npy');rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
x=np.load(R/'features/c10_points/dev_x.npy',mmap_mode='r');y=np.load(D/'dev_y.npy',mmap_mode='r').astype(float);m=np.load(D/'dev_mask.npy',mmap_mode='r').astype(bool)
pred=np.column_stack([np.load(R/f'runs/c10_points_phase_geometry_s20261101/dev_pred{b}.npy') for b in ['415','58']]);r=y-pred
man=json.loads((O/'residual_probe/manifest.json').read_text());calset=set(man['calibration_terrains']);pt=np.repeat([rows[int(i)]['terrain_id'] for i in ids],256)
cal=np.isin(pt,list(calset));conf=~cal;joint=m.all(1)
e3=(2*np.sinh(3*x[:,11:19])).max(1);e2=2*np.sinh(3*x[:,19]);zone=np.where(e3>=0,2,np.where(e2>=0,1,0))
names=['nominal_clear','height_sensitive','nominal_blocked'];results={};new=pred.astype(float).copy()
ct=sorted(set(pt[conf]));code={t:i for i,t in enumerate(ct)};tc=np.array([code[t] for t in pt[conf]])
rng=np.random.default_rng(20261107);w=rng.multinomial(len(ct),np.full(len(ct),1/len(ct)),size=1000)
for z,name in enumerate(names):
 a=cal&joint&(zone==z);q=conf&joint&(zone==z);v=q[conf]
 out={'calibration_joint_pixels':int(a.sum()),'confirmation_joint_pixels':int(q.sum()),
 'calibration_residual_correlation':float(np.corrcoef(r[a].T)[0,1]),'confirmation_residual_correlation':float(np.corrcoef(r[q].T)[0,1]),'bands':{}}
 # Terrain bootstrap of correlation using sufficient statistics.
 u=r[conf,0];vv=r[conf,1];moment=np.stack([np.bincount(tc,weights=aa*v,minlength=len(ct)) for aa in [np.ones(len(tc)),u,vv,u*u,vv*vv,u*vv]],1)
 mom=w@moment;n=mom[:,0];num=mom[:,5]-mom[:,1]*mom[:,2]/n;den=np.sqrt((mom[:,3]-mom[:,1]**2/n)*(mom[:,4]-mom[:,2]**2/n))
 out['confirmation_correlation_terrain_bootstrap95']=np.percentile(num/den,[2.5,97.5]).tolist()
 for k,b in enumerate(['415','58']):
  u=r[a,1-k];vv=r[a,k];slope=float(((u-u.mean())*(vv-vv.mean())).sum()/((u-u.mean())**2).sum());intercept=float(vv.mean()-slope*u.mean())
  fix=intercept+slope*r[:,1-k];new[zone==z,k]+=fix[zone==z];err=new[q,k]-y[q,k];old=r[q,k]
  ss=np.bincount(tc,weights=(new[conf,k]-y[conf,k])**2*v,minlength=len(ct));base=np.bincount(tc,weights=r[conf,k]**2*v,minlength=len(ct))
  out['bands'][b]={'slope':slope,'intercept':intercept,'baseline_RMSE_joint':float(np.sqrt(np.mean(old**2))),'corrected_RMSE_joint':float(np.sqrt(np.mean(err**2))),'SSE_reduction_percent':float(100*(1-np.sum(err**2)/np.sum(old**2))),
   'SSE_reduction_terrain_bootstrap95':np.percentile(100*(1-(w@ss)/(w@base)),[2.5,97.5]).tolist()}
 results[name]=out
out={'scope':'Exploratory diagnostic after global cross-band control; linear coefficients fit on306 calibration terrains and evaluated on306 different confirmation terrains. True other-band label is privileged and unavailable at deployment.',
 'reason':'Global cross-band coefficient has different applicability in height-sensitive and nominal-blocked regions. Conditioning uses only preexisting input geometry.',
 'zones':results}
(O/'conditional_privileged_control.json').write_text(json.dumps(out,indent=2)+'\n');np.save(O/'conditional_privileged_confirmation.npy',new[conf].astype(np.float32))
print(json.dumps(out,indent=2))
