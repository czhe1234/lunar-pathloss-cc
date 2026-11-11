"""CPU geometry hypotheses: no training and no propagation simulation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
s=importlib.util.spec_from_file_location('alignment',Path(__file__).with_name('diagnose_geometry_alignment.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
OUT=a.OUT;start=time.time();selection=json.loads((OUT/'geometry_selection.json').read_text())['items']

def one(item):
 i=item['index_row'];r=a.rows[i];h=a.H[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);e0,d=a.profile(h,tr,tc,a.rr,a.cc)
 y=np.stack([b[i,a.rr,a.cc] for b in a.Y]).astype(np.float64);v=np.stack([np.unpackbits(b[i],bitorder='big').reshape(256,256)[a.rr,a.cc].astype(bool) for b in a.M]);norm=y-20*np.log10(4*np.pi*d[None]/a.wavelength[:,None]);valid=v.all(0)&(np.hypot(a.rr-tr,a.cc-tc)>=8)
 cp=(norm.max(0)<=3)&(norm.min(0)>=-6)&valid;bp=(norm.min(0)>=10)&valid
 profiles={};n=384;t=np.arange(1,n+1,dtype=np.float32)/(n+1)
 terrain=a.interpolate(h,tr+(a.rr-tr)[:,None]*t,tc+(a.cc-tc)[:,None]*t)
 txground=float(h[tr,tc]);rxground=h[a.rr,a.cc]
 for th in (1.,1.5,2.,2.5,3.,3.5):
  for rh in (.5,1.,1.5):profiles[f'tx{th}_rx{rh}']=(terrain-((txground+th)+(rxground[:,None]+rh-txground-th)*t)).max(1)
 for dr in (-1.,-.5,0.,.5,1.):
  for dc in (-1.,-.5,0.,.5,1.):
   if dr==dc==0:continue
   profiles[f'txpos_{dr}_{dc}']=a.profile(h,tr+dr,tc+dc,a.rr,a.cc,n=384)[0]
 # Conservative visibility over receiver footprint, used only as a geometry proxy.
 for radius in (.5,1.):
  es=np.stack([a.profile(h,tr,tc,a.rr,a.cc,rx_shift=(dy,dx),n=384)[0] for dy,dx in [(-radius,-radius),(-radius,radius),(radius,-radius),(radius,radius)]])
  profiles[f'all_corners_visible_{radius}']=es.max(0)
 result={}
 for name,e in profiles.items():
  gc=e<0;safe=e<-.5;result[name]={'cp_n':int(cp.sum()),'cp_hit':int((cp&gc).sum()),'bp_n':int(bp.sum()),'bp_hit':int((bp&~gc).sum()),'safe_n':int((safe&valid).sum()),'safe_both_above6_n':int((safe&valid&(norm.min(0)>6)).sum())}
 return {'cohort':item['cohort'],'index_row':i,'terrain_id':r['terrain_id'],'variants':result}

total={};records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for k,r in enumerate(pool.map(one,selection)):
  records.append(r);a.add(total.setdefault(r['cohort'],{}),r['variants'])
  if (k+1)%16==0:print(json.dumps({'phase':'height_and_footprint','maps':k+1,'seconds':time.time()-start}),flush=True)
for c,variants in total.items():
 for name,s in variants.items():s['balanced_accuracy']=.5*(s['cp_hit']/s['cp_n']+s['bp_hit']/s['bp_n']);s['safe_anomaly_fraction']=s['safe_both_above6_n']/s['safe_n'] if s['safe_n'] else None
out={'scope':'geometry metadata hypotheses, fixed candidates; 64 discovery and 64 confirmation terrains; actual power classes are proxies','variants':total,'per_map':records,'seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};a.write('geometry_height_cells.json',out)
print(json.dumps({'done':True,'seconds':time.time()-start,'top8':{c:sorted([(n,s['balanced_accuracy']) for n,s in v.items()],key=lambda t:-t[1])[:8] for c,v in total.items()}}),flush=True)
