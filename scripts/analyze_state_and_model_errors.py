"""CPU geometry sensitivity and frozen-model conditional errors. No fit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
s=importlib.util.spec_from_file_location('alignment',Path(__file__).with_name('diagnose_geometry_alignment.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
start=time.time();z=np.load(a.OUT/'development_geometry_samples.npz');ids=z['original_rows'];Y=z['labels'].astype(np.float64);P=z['predictions'].astype(np.float64);V=z['valid'];E3=z['E'];D=z['distance'];COS=z['cosine'];ref=20*np.log10(4*np.pi*D[:,None,:]/a.wavelength[None,:,None]);YL=Y-ref;PL=P-ref;err=P-Y

def one(i):
 r=a.rows[int(i)];h=a.H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);return a.profile(h,tr,tc,a.rr,a.cc,tx_height=2.,n=384)[0]
with ThreadPoolExecutor(max_workers=4) as pool:E2=np.stack(list(pool.map(one,ids)))
np.save(a.OUT/'development_E2.npy',E2)
zones={'ground3_clear':E3<-.25,'ground3_critical':np.abs(E3)<=.25,'ground3_blocked':E3>.25,'ground3_clear_but_ground2_not_clear':(E3<-.25)&(E2>=-.25),'both_clear':(E3<-.25)&(E2<-.25),'both_blocked':(E3>.25)&(E2>.25),'ground3_strongclear':E3<-.5,'ground3_strongclear_but_ground2_not_clear':(E3<-.5)&(E2>=-.25),'strongclear_highcos':(E3<-.75)&(COS>.03),'strict_clear':(E2<-.5)&(COS>.03)}
result={}
for k,b in enumerate(('415','58')):
 totaln=int(V[:,k].sum());totalsse=float((err[:,k][V[:,k]]**2).sum());result[b]={}
 for name,q0 in zones.items():
  for suffix,q1 in [('',q0),('_label_excess_gt6',q0&(YL[:,k]>6))]:
   q=q1&V[:,k];e=err[:,k][q];y=YL[:,k][q];p=PL[:,k][q];sy=int(q.sum());result[b][name+suffix]={'n':sy,'pixel_share':sy/totaln,'sse_share':float(e@e)/totalsse,'rmse':float(np.sqrt(np.mean(e*e))) if sy else None,'mean_error':float(e.mean()) if sy else None,'label_mean':float(y.mean()) if sy else None,'prediction_mean':float(p.mean()) if sy else None}
 # Descriptive state mistakes, exactly partition all valid SSE.
 sy=np.digitize(YL[:,k],[3.,10.]);sp=np.digitize(PL[:,k],[3.,10.]);result[b]['state_partition']={}
 for name,q0 in [('low_both',(sy==0)&(sp==0)),('same_high',(sy==2)&(sp==2)),('same_middle',(sy==1)&(sp==1)),('different_state',sy!=sp)]:
  q=q0&V[:,k];e=err[:,k][q];result[b]['state_partition'][name]={'pixel_share':float(q.sum())/totaln,'sse_share':float(e@e)/totalsse,'rmse':float(np.sqrt(np.mean(e*e)))}
# B0/B1 on matched saved dev samples, no repeated forward.
f=a.OUT/'frozen_predictions';fi=json.loads((f/'indices.json').read_text());jm={int(i):j for j,i in enumerate(ids)};sub=[j for j,p in enumerate(fi['parts']) if p=='dev'];oldj=np.array([jm[fi['index_rows'][j]] for j in sub]);vv=V[oldj];yy=Y[oldj];base=ref[oldj];substates=np.digitize(yy-base,[3.,10.]);frozen={}
for name in ('b0_bf16','b1_bf16'):
 pa=np.load(f/(name+'.npy'),mmap_mode='r');pp=np.stack([pa[j,:,a.rr,a.cc].T for j in sub]).astype(np.float64);assert pp.shape==yy.shape;ee=pp-yy;sp=np.digitize(pp-base,[3.,10.]);frozen[name]={}
 for k,b in enumerate(('415','58')):
  n=int(vv[:,k].sum());sse=float((ee[:,k][vv[:,k]]**2).sum());r={}
  for nm,qq in [('all',np.ones_like(E3[oldj],bool)),('low_both',(substates[:,k]==0)&(sp[:,k]==0)),('same_high',(substates[:,k]==2)&(sp[:,k]==2)),('different_state',substates[:,k]!=sp[:,k]),('both_geom_clear',zones['both_clear'][oldj]),('both_geom_blocked',zones['both_blocked'][oldj])]:
   q=qq&vv[:,k];e=ee[:,k][q];r[nm]={'pixel_share':float(q.sum())/n,'sse_share':float(e@e)/sse,'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean())}
  frozen[name][b]=r
# A single preset diagnostic clipping rule (no thresholds chosen by validation score).
# Existing predictions only: high-confidence geometry cannot justify arbitrary large loss.
# This is deliberately a bounded check, not a new simulation or supervised fit.
selection=json.loads((a.OUT/'geometry_selection.json').read_text())['items'];cohorts={x['index_row']:x['cohort'] for x in selection};rule={}
for cohort in ('discovery','confirmation'):
 keep=np.array([cohorts.get(int(i))==cohort for i in ids]);rule[cohort]={}
 for k,b in enumerate(('415','58')):
  v=V[keep,k];e=err[keep,k];g=zones['strict_clear'][keep];p=P[keep,k];y=Y[keep,k];r=ref[keep,k];pnew=np.where(g,np.minimum(p,r),p);en=pnew-y
  rule[cohort][b]={'n':int(v.sum()),'baseline_rmse':float(np.sqrt(np.mean(e[v]**2))),'rule_rmse':float(np.sqrt(np.mean(en[v]**2))),'changed_pixel_fraction':float((g&(p>r)&v).sum())/int(v.sum()),'note':'not official validation; input-only gate; no new propagation training labels'}
a.write('state_and_model_errors.json',{'scope':'all2562devmaps/fixed1024pixels; no training, original3m metadata retained; 2m used only as geometry sensitivity','groups':result,'matched_frozen':frozen,'preset_clipping_diagnostic':rule,'seconds':time.time()-start})
print(json.dumps({'done':True,'seconds':time.time()-start,'state_partition':{b:r['state_partition'] for b,r in result.items()},'matched_frozen':frozen,'rule':rule}),flush=True)
