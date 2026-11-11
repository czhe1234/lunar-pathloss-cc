"""Local label structure around sampled anomalies; no smoothing of training targets."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
s=importlib.util.spec_from_file_location('alignment',Path(__file__).with_name('diagnose_geometry_alignment.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
z=np.load(a.OUT/'development_geometry_samples.npz');ids=z['original_rows'];E=z['E'];COS=z['cosine'];P=z['predictions'].astype(np.float64);YY=z['labels'].astype(np.float64);V=z['valid'];er=P-YY;start=time.time();o=np.arange(-4,5);dr,dc=np.meshgrid(o,o,indexing='ij');rr=np.clip(a.rr[:,None]+dr.ravel(),0,255);cc=np.clip(a.cc[:,None]+dc.ravel(),0,255)
def one(j):
 i=int(ids[j]);r=a.rows[i];h=a.H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);d=np.sqrt((rr-tr)**2+(cc-tc)**2+(h[rr,cc]+1-h[tr,tc]-3)**2);out={}
 for k,b in enumerate(('415','58')):
  vv=np.unpackbits(a.M[k][i],bitorder='big').reshape(256,256)[rr,cc].astype(bool);yy=a.Y[k][i,rr,cc].astype(np.float64)-20*np.log10(4*np.pi*d/a.wavelength[k]);center=yy[:,40];lo=(vv&(yy<=3)).sum(1)/np.maximum(vv.sum(1),1);anomaly=V[j,k]&(E[j]<-.5)&(center>6);isolated=anomaly&(lo>=.75);out[b]={}
  for nm,q in [('all_valid',V[j,k]),('strongclear_anomaly',anomaly),('anomaly_low_neighbors75pct',isolated),('anomaly_other',anomaly&~isolated),('anomaly_low_cos',anomaly&(COS[j]<.01))]:
   ee=er[j,k][q];out[b][nm]={'n':int(q.sum()),'sse':float(ee@ee)}
 return out
summary={}
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(one,range(len(ids))):a.add(summary,r)
for b,groups in summary.items():
 for name,r in groups.items():r['pixel_share']=r['n']/groups['all_valid']['n'];r['sse_share']=r['sse']/groups['all_valid']['sse'];r['rmse']=(r['sse']/r['n'])**.5 if r['n'] else None
a.write('local_label_structure.json',{'scope':'fixed1024receivers on2562devmaps; 9x9neighborhood read from original valid labels, never used as new training targets','groups':summary,'limits':'isolated high attenuation is an observation pattern, not proof of random noise or an irreducible error floor; original mesh, paths and independent generator repeats are absent','seconds':time.time()-start});print(json.dumps({'done':True,'seconds':time.time()-start,'groups':summary}),flush=True)
