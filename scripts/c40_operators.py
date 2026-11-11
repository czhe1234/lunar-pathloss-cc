"""RT operator and Monte Carlo checks on one FIT map. No model training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import time,json,csv
import numpy as np
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
R=Path(__file__).resolve().parents[1];O=R/'audit/c40_rt_pilot_20261101';D=R.parent/'data'
d=O/'map0_floor_1024x4000000'
while not (d/'result.json').exists():time.sleep(3)
cfg=json.loads((d/'config.json').read_text());idx=cfg['index']
rows=list(csv.DictReader((D/'train_index.csv').open()));row=rows[idx]
h=np.array(np.load(D/'train_hm.npy',mmap_mode='r')[int(row['hm_row'])],np.float32)
yy,xx=np.mgrid[:256,:256]
v=np.stack([xx,yy,h],-1).astype(np.float32).reshape(-1,3);base=(yy[:-1,:-1]*256+xx[:-1,:-1]).reshape(-1)
faces=np.stack([np.stack([base,base+1,base+257],1),np.stack([base,base+257,base+256],1)],1).reshape(-1,3)
centers=v[faces].mean(1)[:,:2]
gg=np.array(np.load(R/'features/c3/train.npy',mmap_mode='r')[idx],np.float32)
e3=np.max(2*np.sinh(3*gg[:8]),axis=0);e2=2*np.sinh(3*gg[8])
zone=np.where(e2<-.5,0,np.where(e3>2,2,1))
interior=(xx>=1)&(xx<=254)&(yy>=1)&(yy<=254)
while not (O/f'C37_train_map{idx}.npy').exists():time.sleep(3)
baseline=np.load(O/f'C37_train_map{idx}.npy')
def interp(p,mode):
 active=p>0 if mode!='power_with_zeros' else np.ones(len(p),bool)
 values=(-10*np.log10(p[active])) if mode=='db_positive' else p[active]
 out=LinearNDInterpolator(centers[active],values)(xx,yy)
 missing=~np.isfinite(out)
 if missing.any():out[missing]=NearestNDInterpolator(centers[active],values)(xx[missing],yy[missing])
 return out if mode=='db_positive' else -10*np.log10(np.maximum(out,1e-30))
records=[]
for k,b in enumerate(['415','58']):
 y=np.array(np.load(D/f'train_pl{b}.npy',mmap_mode='r')[idx],float)
 mask=np.unpackbits(np.load(D/f'train_mask{b}.npy',mmap_mode='r')[idx],bitorder='big').reshape(256,256).astype(bool)&interior
 pg=np.load(d/f'face_power{b}.npy');halves=np.load(d/f'power_halves{b}.npz')
 for mode in ['C37','power_with_zeros','power_positive','db_positive']:
  pred=baseline[k] if mode=='C37' else interp(pg,mode)
  assert np.isfinite(pred[mask]).all()
  pred=np.clip(pred,0,180)
  for zi,zn in [(-1,'all'),(0,'safe_LOS'),(1,'boundary'),(2,'occluded')]:
   use=mask if zi==-1 else mask&(zone==zi)
   err=pred[use]-y[use]
   rec={'band':b,'operator':mode,'zone':zn,'n':int(use.sum()),'RMSE':float(np.sqrt(np.mean(err**2))) if len(err) else None,'bias':float(err.mean()) if len(err) else None}
   records.append(rec)
  if mode!='C37':
   p1,p2=[np.clip(interp(halves[n],mode),0,180) for n in ['odd','even']]
   e=p1[mask]-p2[mask];stable=mask&(abs(p1-p2)<.5)
   records.append({'band':b,'operator':mode,'half_difference_RMSE':float(np.sqrt(np.mean(e**2))),'stable_lt_half_db_valid_fraction':float(stable.sum()/mask.sum()),'label_RMSE_on_stable_diagnostic':float(np.sqrt(np.mean((pred[stable]-y[stable])**2)))})
  np.save(O/f'operator_{b}_{mode}.npy',pred.astype(np.float32))
out={'scope':'One FIT terrain/map; candidate input reconstruction and observation operators only; not DEV or official score','index':idx,'rays_per_band':4096000000,'records':records,'limitations':'Positive-cell interpolation uses nearest positive extrapolation outside its convex hull; no target/mask-based filling. Original mesh/TX/interpolation not recovered. Half-difference quantifies Monte Carlo sensitivity, not missing-input error floor.'}
(O/'operator_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
