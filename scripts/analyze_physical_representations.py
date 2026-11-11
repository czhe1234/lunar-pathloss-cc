"""CPU-only descriptive error audit. No training, GPU use, or propagation simulation.
Uses pre-existing B0 predictions and original training-split data only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv,json,time,hashlib,datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';OUT=ROOT/'audit/physical_representations_20261031';OUT.mkdir(exist_ok=True)
PRED=ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16'
ids=json.loads((PRED/'indices.json').read_text());assert ids['source_split']=='train';indices=ids['index_rows']
rows=list(csv.DictReader((DATA/'train_index.csv').open()));dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids']);assert all(rows[i]['terrain_id'] in dev for i in indices)
hm=np.load(DATA/'train_hm.npy',mmap_mode='r',allow_pickle=False)
y=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')];packed=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')];pred=[np.load(PRED/f'pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')]
yr,xc=np.meshgrid(np.arange(4,256,8),np.arange(4,256,8),indexing='ij');yr=yr.flatten();xc=xc.flatten();P=len(yr);t=np.arange(1,193,dtype=np.float32)/193
meta=json.loads((DATA/'metadata.json').read_text());freq_gap=20*np.log10(meta['bands_hz']['58']/meta['bands_hz']['415'])
for name,ref in json.loads((ROOT/'audit/raw_audit.json').read_text())['files'].items():
 st=(DATA/name).stat();assert st.st_size==ref['size_bytes'] and st.st_mtime_ns==ref['mtime_ns']

def moments(a):
 a=np.asarray(a,dtype=np.float64);return [len(a),float(a.sum()),float(np.square(a).sum())]
def audit_one(j):
 i=indices[j];row=rows[i];h=hm[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col']);dx=xc.astype(np.float32)-tc;dy=yr.astype(np.float32)-tr;horizontal=np.hypot(dx,dy)
 txz=h[tr,tc]+3.;rxz=h[yr,xc]+1.
 gx=tc+dx[:,None]*t;gy=tr+dy[:,None]*t
 ht=map_coordinates(h,np.stack((gy,gx)),order=1,mode='nearest',prefilter=False)
 line=txz+(rxz[:,None]-txz)*t;excess=ht-line;emax=excess.max(1);peak_t=t[excess.argmax(1)]
 zones={'clear_margin':emax<-.25,'near_zero_clearance':np.abs(emax)<=.25,'blocked_margin':emax>.25}
 ranges={'range_lt64':horizontal<64,'range_64to128':(horizontal>=64)&(horizontal<128),'range_ge128':horizontal>=128}
 slope_x=(h[yr,xc+1]-h[yr,xc-1])*.5;slope_y=(h[yr+1,xc]-h[yr-1,xc])*.5
 dz=rxz-txz;distance=np.sqrt(dx*dx+dy*dy+dz*dz)
 cos_angle=np.abs(slope_x*dx+slope_y*dy-dz)/np.sqrt(1+slope_x*slope_x+slope_y*slope_y)/distance
 groups={'all':np.ones(P,dtype=bool),**zones,**ranges,'surface_grazing_cos_lt0p1':cos_angle<.1,'surface_cos_ge0p1':cos_angle>=.1}
 for zn,zm in zones.items():
  for rn,rm in ranges.items():groups[zn+'__'+rn]=zm&rm
 for name,condition in [('tx_third',peak_t<1/3),('middle_third',(peak_t>=1/3)&(peak_t<2/3)),('rx_third',peak_t>=2/3)]:groups['blocked_peak_'+name]=zones['blocked_margin']&condition
 truths=np.stack([a[i,yr,xc].astype(np.float64) for a in y]);estimates=np.stack([a[j,yr,xc].astype(np.float64) for a in pred]);errors=estimates-truths
 wavelengths=np.array([299792458./meta['bands_hz'][b] for b in ('415','58')])
 # Dimensionless, distance-normalized measured gain. This transforms released labels; no simulated target is generated.
 normalized_gain_db=-truths+20*np.log10(4*np.pi*distance[None,:]/wavelengths[:,None])
 masks=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256).astype(bool) for a in packed]);valid=masks[:,yr,xc]
 band_stats={}
 for b,key in enumerate(('415','58')):
  statistics={}
  for name,select in groups.items():
   q=select&valid[b];e=errors[b,q];z=normalized_gain_db[b,q];statistics[name]={'count':int(q.sum()),'sse':float(np.square(e).sum()),'signed_error_sum':float(e.sum()),'maps':int(q.any()),'normalized_gain_db_sum':float(z.sum()),'normalized_gain_db_square_sum':float((z*z).sum()),'normalized_gain_within_1db_count':int((np.abs(z)<=1).sum()),'normalized_gain_within_0p5db_count':int((np.abs(z)<=.5).sum())}
  center=truths[b];neighbor_values=np.stack([y[b][i,yr-1,xc],y[b][i,yr+1,xc],y[b][i,yr,xc-1],y[b][i,yr,xc+1]]).astype(np.float64)
  neighbor_valid=masks[b,yr-1,xc]&masks[b,yr+1,xc]&masks[b,yr,xc-1]&masks[b,yr,xc+1]
  jump=np.max(np.abs(neighbor_values-center),axis=0)>5
  diagnostic={'label_jump_gt5_db':neighbor_valid&jump,'label_jump_le5_db':neighbor_valid&~jump,'label_neighbor_invalid':~neighbor_valid}
  for name,select in diagnostic.items():
   q=select&valid[b];e=errors[b,q];statistics[name]={'count':int(q.sum()),'sse':float(np.square(e).sum()),'signed_error_sum':float(e.sum()),'maps':int(q.any())}
   q &= horizontal>=16;e=errors[b,q];statistics[name+'__range_ge16']={'count':int(q.sum()),'sse':float(np.square(e).sum()),'signed_error_sum':float(e.sum()),'maps':int(q.any())}
  band_stats[key]=statistics
 both=valid[0]&valid[1];pair={}
 for name,select in {'all':groups['all'],**zones}.items():
  q=select&both;delta=truths[1,q]-truths[0,q];a=errors[0,q];b=errors[1,q]
  pair[name]={'delta':moments(delta),'frequency_adjusted_delta':moments(delta-freq_gap),'error415':moments(a),'error58':moments(b),'error_product_sum':float((a*b).sum()),'label415':moments(truths[0,q]),'label58':moments(truths[1,q]),'label_product_sum':float((truths[0,q]*truths[1,q]).sum()),'normalized415':moments(normalized_gain_db[0,q]),'normalized58':moments(normalized_gain_db[1,q]),'normalized_product_sum':float((normalized_gain_db[0,q]*normalized_gain_db[1,q]).sum())}
 return {'sample_id':row['sample_id'],'terrain_id':row['terrain_id'],'group':row['group'],'band_stats':band_stats,'pair':pair}

def add(a,b):
 if isinstance(b,dict):
  for k,v in b.items():
   if k not in a:a[k]=v
   elif isinstance(v,dict):add(a[k],v)
   elif isinstance(v,list):a[k]=[x+y for x,y in zip(a[k],v)]
   else:a[k]+=v
 return a

start=time.time();totals={};pairs={};permap=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for j,result in enumerate(pool.map(audit_one,range(len(indices)))):
  # Copy per-map numbers before aggregation mutates the first dictionary.
  permap.append({'sample_id':result['sample_id'],'terrain_id':result['terrain_id'],'group':result['group'],**{f'{b}_{g}_{k}':v[k] for b,gs in result['band_stats'].items() for g,v in gs.items() if g in ('all','clear_margin','near_zero_clearance','blocked_margin') for k in ('count','sse')}})
  add(totals,result['band_stats']);add(pairs,result['pair'])
  if (j+1)%200==0:print(json.dumps({'completed_maps':j+1,'total':len(indices),'seconds':time.time()-start}),flush=True)
for b,gs in totals.items():
 overall=gs['all']
 for name,s in gs.items():
  n=s['count'];s['rmse_db']=float(np.sqrt(s['sse']/n)) if n else None;s['mean_signed_error_db']=s['signed_error_sum']/n if n else None
  s['share_of_valid_sampled_pixels']=n/overall['count'];s['share_of_sampled_sse']=s['sse']/overall['sse']
  if 'normalized_gain_db_sum' in s and n:
   s['normalized_gain_db_mean']=s['normalized_gain_db_sum']/n;s['normalized_gain_db_std']=float(np.sqrt(max(0,s['normalized_gain_db_square_sum']/n-s['normalized_gain_db_mean']**2)));s['normalized_gain_db_rms']=float(np.sqrt(s['normalized_gain_db_square_sum']/n));s['normalized_gain_within_1db_fraction']=s['normalized_gain_within_1db_count']/n;s['normalized_gain_within_0p5db_fraction']=s['normalized_gain_within_0p5db_count']/n
  s['remaining_rmse_if_this_group_error_were_zero']=float(np.sqrt(max(0,overall['sse']-s['sse'])/overall['count']))

def mean_std(m):
 n,s,q=m;return {'n':int(n),'mean':s/n if n else None,'std':float(np.sqrt(max(0,q/n-(s/n)**2))) if n else None}
for name,s in pairs.items():
 for prefix in ['error','label','normalized']:
  a=s[prefix+'415'];b=s[prefix+'58'];n=a[0];cov=s[prefix+'_product_sum']/n-(a[1]/n)*(b[1]/n) if n else 0
  va=max(0,a[2]/n-(a[1]/n)**2) if n else 0;vb=max(0,b[2]/n-(b[1]/n)**2) if n else 0
  s[prefix+'_correlation']=float(cov/np.sqrt(va*vb)) if va*vb>0 else None
 for key in ['delta','frequency_adjusted_delta','error415','error58','label415','label58','normalized415','normalized58']:s[key]=mean_std(s[key])
report={'analysis_type':'CPU-only descriptive audit of existing B0 errors; no training or new propagation data','model_run':'b0_raw_s20261031','checkpoint_epoch':60,'source_split':'internal development subset of train','maps':len(indices),'terrains':len({rows[i]['terrain_id'] for i in indices}),'sampling':{'pixels_per_map':P,'grid_stride_pixels':8,'grid_offset_pixels':4,'profile_internal_samples':192,'clearance_group_threshold_m':.25,'label_jump_threshold_db_per_neighbor':5,'workers':4,'uses_gpu':False},'frequency_gap_ideal_free_space_db':float(freq_gap),'bands':totals,'joint_valid_band_statistics':pairs,'limitations':['Subsampled pixel statistics, not official or full-image validation scores.','Geometry is a bilinear heightmap segment test, not an exact reconstruction of the simulation mesh.','Clearance groups describe direct geometric visibility, not validity masks, true individual path powers or exhaustive LOS/NLOS classification.','Label jumps are diagnostic supervision-side quantities; they are not available as test inputs.','Zeroing a group of observed errors is error accounting, not an achievable model result or an irreducible lower bound.','Two-band correlations do not establish causal attribution; no independent Monte Carlo replicates are available.'],'seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'error_structure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
with (OUT/'per_map_strata.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(permap[0]));w.writeheader();w.writerows(permap)
print(json.dumps({'done':True,'seconds':report['seconds'],'output':str(OUT/'error_structure.json'),'sampled_rmse':{b:totals[b]['all']['rmse_db'] for b in ['415','58']}}),flush=True)
