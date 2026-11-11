"""CPU descriptive checks for receiver terrain structure and same-terrain TX errors.
All label-derived means are diagnostics. No predictor, calibration, or training.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,importlib.util
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter,laplace
spec=importlib.util.spec_from_file_location('base_audit',Path(__file__).with_name('attribute_physical_errors.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=m.OUT;start=time.time()
@lru_cache(maxsize=700)
def terrain_fields(hm_row):
 h=np.asarray(m.hm[hm_row],np.float64);sy,sx=np.gradient(h)
 residual4=h-uniform_filter(h,size=9,mode='reflect');rough4=np.sqrt(np.maximum(0,uniform_filter(residual4**2,size=9,mode='reflect')))
 residual16=h-uniform_filter(h,size=33,mode='reflect');rough16=np.sqrt(np.maximum(0,uniform_filter(residual16**2,size=33,mode='reflect')))
 return sx,sy,rough4,rough16,laplace(h,mode='reflect')
def one(j):
 i=m.indices[j];r=m.rows[i];h=m.hm[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col'])
 Y=np.stack([a[i,m.yr,m.xc] for a in m.y]).astype(np.float64);P=np.stack([a[j,m.yr,m.xc] for a in m.pred]).astype(np.float64)
 V=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256)[m.yr,m.xc].astype(bool) for a in m.packed]);both=V.all(0);error=P-Y
 E=m.profile(h,tr,tc,m.yr,m.xc);zone=np.where(E<-.25,0,np.where(E>.25,2,1));horizontal=np.hypot(m.xc-tc,m.yr-tr);base=zone*3+np.digitize(horizontal,[64,128])
 sx,sy,r4,r16,curv=[a[m.yr,m.xc] for a in terrain_fields(int(r['hm_row']))]
 dx=m.xc-tc;dy=m.yr-tr;dz=h[m.yr,m.xc]+1.-(float(h[tr,tc])+3.)
 cosine=np.abs(-sx*dx-sy*dy+dz)/np.sqrt(1+sx*sx+sy*sy)/np.sqrt(dx*dx+dy*dy+dz*dz)
 descriptors={'receiver_angle':np.digitize(cosine,[.02,.05,.1,.25,.5]),'receiver_slope':np.digitize(np.hypot(sx,sy),[.02,.05,.1,.25,.5,1.]),'roughness_4m':np.digitize(r4,[.05,.1,.25,.5,1.,2.]),'roughness_16m':np.digitize(r16,[.1,.25,.5,1.,2.,5.]),'curvature':np.digitize(curv,[-.1,-.02,.02,.1])}
 partitions={'zone_range':base}
 for name,b in descriptors.items():partitions['zone_range_'+name]=base*8+b
 partitions['zone_range_angle_roughness']=base*64+descriptors['receiver_angle']*8+descriptors['roughness_4m']
 return {name:{str(int(k)):m.pair_stat(error,Y,(bins==k)&both) for k in np.unique(bins)} for name,bins in partitions.items()}

totals={}
with ThreadPoolExecutor(max_workers=4) as pool:
 for k,r in enumerate(pool.map(one,range(len(m.indices)))):
  m.add(totals,r)
  if (k+1)%400==0:print(json.dumps({'phase':'terrain_descriptors','maps':k+1,'seconds':time.time()-start}),flush=True)
summary={}
for name,p in totals.items():
 n=sum(s['n'] for s in p.values());ds=sum(s['delta_sum'] for s in p.values());d2=sum(s['delta_square_sum'] for s in p.values());between=sum(s['delta_sum']**2/s['n'] for s in p.values() if s['n'])-ds*ds/n
 summary[name]={'n':n,'nonempty_groups':sum(s['n']>0 for s in p.values()),'delta_between_group_variance_share':between/(d2-ds*ds/n),**{b+'_group_mean_error_sse_share':sum(s['error'+b+'_sum']**2/s['n'] for s in p.values() if s['n'])/sum(s['sse'+b] for s in p.values()) for b in ('415','58')}}
print(json.dumps({'phase':'descriptors_done','summary':summary}),flush=True)
# Same-terrain, same-receiver error mean over actual distinct TX, restricted to >=10 valid TX.
terrains={}
for j,i in enumerate(m.indices):
 r=m.rows[i]
 if r['group']=='multi':terrains.setdefault(r['terrain_id'],[]).append(j)
def multi_one(item):
 terrain,js=item;assert len(js)==51
 result={}
 for b,key in enumerate(('415','58')):
  n=np.zeros((256,256),np.int64);se=np.zeros((256,256),np.float64);sse=np.zeros((256,256),np.float64)
  for j in js:
   i=m.indices[j];e=np.asarray(m.pred[b][j],np.float64)-np.asarray(m.y[b][i],np.float64);v=np.unpackbits(m.packed[b][i],bitorder='big').reshape(256,256).astype(bool)
   n+=v;se+=np.where(v,e,0);sse+=np.where(v,e*e,0)
  eligible=n>=10;mean_component=float((se[eligible]**2/n[eligible]).sum())
  result[key]={'all_valid_n':int(n.sum()),'all_sse':float(sse.sum()),'eligible_valid_n':int(n[eligible].sum()),'eligible_sse':float(sse[eligible].sum()),'fixed_receiver_mean_component_sse':mean_component,'eligible_receivers':int(eligible.sum())}
 return {'terrain_id':terrain,'values':result}
multi=[];aggregate={}
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(multi_one,terrains.items()):multi.append(r);m.add(aggregate,r['values'])
for b,s in aggregate.items():
 s['eligible_pixel_share']=s['eligible_valid_n']/s['all_valid_n'];s['mean_component_share_of_eligible_sse']=s['fixed_receiver_mean_component_sse']/s['eligible_sse'];s['baseline_eligible_rmse_db']=float(np.sqrt(s['eligible_sse']/s['eligible_valid_n']));s['oracle_remaining_eligible_rmse_db']=float(np.sqrt((s['eligible_sse']-s['fixed_receiver_mean_component_sse'])/s['eligible_valid_n']))
 assert 0<=s['mean_component_share_of_eligible_sse']<=1
out={'scope':{'uses_gpu':False,'model_fitting':False,'sampled_maps':len(m.indices),'sampled_receivers_per_map':len(m.yr),'multi_terrains':len(terrains),'multi_tx_each':51,'descriptor_stats':'descriptive conditional means on joint-valid sampled receivers; not held-out predictive performance','multi_stats':'all valid full-resolution receivers on 39 development terrains, retaining receivers with >=10 valid TX; label-dependent decomposition only','normal_angle':'receiver surface normal versus direct TX-RX segment, a geometry descriptor, not antenna gain','roughness':'RMS height residual after 9x9 or 33x33 local averaging, measured in meters; macro terrain only','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'descriptors':summary,'partitions':totals,'multi_tx_error_structure':aggregate,'multi_terrain_details':multi,'seconds':time.time()-start}
(OUT/'remaining_descriptors.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'done':True,'seconds':time.time()-start,'multi_tx':aggregate}),flush=True)
