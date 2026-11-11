"""Alternative geometric state descriptions and threshold sensitivity; no learning."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
s=importlib.util.spec_from_file_location('alignment',Path(__file__).with_name('diagnose_geometry_alignment.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
start=time.time();selection=json.loads((a.OUT/'geometry_selection.json').read_text())['items'];allcache=np.load(a.OUT/'development_geometry_samples.npz');jmap={int(i):j for j,i in enumerate(allcache['original_rows'])}
def one(item):
 i=item['index_row'];r=a.rows[i];h=a.H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);e3,d=a.profile(h,tr,tc,a.rr,a.cc,n=512);e2,_=a.profile(h,tr,tc,a.rr,a.cc,tx_height=2.,n=512);em,_=a.profile(h,tr,tc,a.rr,a.cc,tx_height=2.,rx_height=0.,n=512)
 y=np.stack([b[i,a.rr,a.cc] for b in a.Y]).astype(np.float64);v=np.stack([np.unpackbits(b[i],bitorder='big').reshape(256,256)[a.rr,a.cc].astype(bool) for b in a.M]);norm=y-20*np.log10(4*np.pi*d[None]/a.wavelength[:,None]);valid=v.all(0)&(np.hypot(a.rr-tr,a.cc-tc)>=8)
 sy,sx=np.gradient(h);signed=(sx[a.rr,a.cc]*(a.cc-tc)+sy[a.rr,a.cc]*(a.rr-tr)-(h[a.rr,a.cc]+1.-h[tr,tc]-3.))/np.sqrt(1+sx[a.rr,a.cc]**2+sy[a.rr,a.cc]**2)/d
 modes={'ground3':e3<0,'ground2':e2<0,'measurement_first_hit':em<0,'ground3_and_front':(e3<0)&(signed>0),'ground3_safe0.5':e3<-.5};res={}
 for threshold in (3.,6.,10.,15.,20.):
  cp=(norm.max(0)<=3)&(norm.min(0)>=-6)&valid;bp=(norm.min(0)>=threshold)&valid
  for name,gc in modes.items():res[f'{name}_blocked_ge{threshold}']={'cp_n':int(cp.sum()),'cp_hit':int((cp&gc).sum()),'bp_n':int(bp.sum()),'bp_hit':int((bp&~gc).sum())}
 for name,g in {'front':signed>0,'back':signed<=0}.items():
  q=valid&(e3<-.5)&g;res[name]={'n':int(q.sum()),'both_excess_gt6':int((q&(norm.min(0)>6)).sum())}
 return {'cohort':item['cohort'],'index_row':i,'variants':res}
totals={};records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(one,selection):records.append(r);a.add(totals.setdefault(r['cohort'],{}),r['variants'])
for variants in totals.values():
 for name,r in variants.items():
  if 'cp_n' in r:r['ba']=.5*(r['cp_hit']/r['cp_n']+r['bp_hit']/r['bp_n'])
a.write('state_definition.json',{'scope':'power-based state proxies, threshold sensitivity on same128fixedterrains; measurement first-hit is an unverified observation hypothesis','variants':totals,'per_map':records,'seconds':time.time()-start})
print(json.dumps({'done':True,'seconds':time.time()-start,'variants':totals}),flush=True)
