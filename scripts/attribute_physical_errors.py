"""Independent CPU-only diagnostics of saved CC B0 predictions. No model fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,json,time,hashlib,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt, minimum_filter, maximum_filter
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';OUT=ROOT/'audit/physical_error_attribution_20261031';OUT.mkdir(exist_ok=True)
PRED=ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16'
rows=list(csv.DictReader((DATA/'train_index.csv').open()))
ids=json.loads((PRED/'indices.json').read_text());assert ids['source_split']=='train';indices=ids['index_rows']
meta=json.loads((DATA/'metadata.json').read_text());gap=float(20*np.log10(meta['bands_hz']['58']/meta['bands_hz']['415']))
hm=np.load(DATA/'train_hm.npy',mmap_mode='r',allow_pickle=False)
y=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')]
pred=[np.load(PRED/f'pl{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')]
packed=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r',allow_pickle=False) for b in ('415','58')]
for name,ref in json.loads((ROOT/'audit/raw_audit.json').read_text())['files'].items():
 st=(DATA/name).stat();assert st.st_size==ref['size_bytes'] and st.st_mtime_ns==ref['mtime_ns']
yr,xc=np.meshgrid(np.arange(4,256,8),np.arange(4,256,8),indexing='ij');yr=yr.ravel();xc=xc.ravel()

def maps(j):
 i=indices[j];r=rows[i]
 Y=np.stack([a[i] for a in y]).astype(np.float64);P=np.stack([a[j] for a in pred]).astype(np.float64)
 V=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256).astype(bool) for a in packed])
 return r,hm[int(r['hm_row'])],Y,P,V

def profile(h,tr,tc,rr,cc,n=192,detail=False):
 t=np.arange(1,n+1,dtype=np.float32)/(n+1);txz=float(h[tr,tc])+3.
 rr=np.asarray(rr,np.float32);cc=np.asarray(cc,np.float32)
 rxz=map_coordinates(h,np.stack((rr,cc)),order=1,mode='nearest',prefilter=False)+1.
 gy=tr+(rr-tr)[:,None]*t;gx=tc+(cc-tc)[:,None]*t
 excess=map_coordinates(h,np.stack((gy,gx)),order=1,mode='nearest',prefilter=False)-(txz+(rxz[:,None]-txz)*t)
 mx=excess.max(1)
 if not detail:return mx
 # Both endpoints are above the terrain; append their known negative signs.
 above=np.pad(excess>0,((0,0),(1,1)),constant_values=False)
 crossings=np.count_nonzero(above[:,1:]!=above[:,:-1],axis=1)
 peak=excess.argmax(1);peak_t=t[peak]
 # Estimate the LAST actual surface crossing, not the obstruction maximum.
 # At an interior maximum, the ray is tangent to the profile; its angle is not an interaction angle.
 aug=np.concatenate((np.full((len(rr),1),-3.,np.float32),excess,np.full((len(rr),1),-1.,np.float32)),axis=1)
 k=np.arange(len(rr));last=np.max(np.where(aug>0,np.arange(n+2)[None,:],0),axis=1)
 left=aug[k,last];right=aug[k,last+1];alpha=np.divide(left,left-right,out=np.zeros_like(left),where=np.abs(left-right)>1e-8)
 crossing_t=(last+np.clip(alpha,0,1))/(n+1)
 hy,hx=np.gradient(h);px=tc+(cc-tc)*crossing_t;py=tr+(rr-tr)*crossing_t
 sx=map_coordinates(hx,np.stack((py,px)),order=1,mode='nearest',prefilter=False)
 sy=map_coordinates(hy,np.stack((py,px)),order=1,mode='nearest',prefilter=False)
 dx=cc-tc;dy=rr-tr;dz=rxz-txz;length=np.sqrt(dx*dx+dy*dy+dz*dz)
 cosine=np.abs(-sx*dx-sy*dy+dz)/(np.sqrt(1+sx*sx+sy*sy)*length)
 cosine=np.where(crossings>0,cosine,0.)
 return mx,crossings,peak_t,cosine

def stat(e,mask):
 z=e[mask];return {'n':int(mask.sum()),'sse':float(np.square(z).sum()),'sum_error':float(z.sum())}

def pair_stat(e,Y,mask):
 a=e[0,mask];b=e[1,mask];c=(a+b)/2;d=(b-a)/2;delta=Y[1,mask]-Y[0,mask]-gap
 s={'n':len(a),'sse415':float((a*a).sum()),'sse58':float((b*b).sum()),'common_sse':float((c*c).sum()),'difference_sse':float((d*d).sum()),'error415_sum':float(a.sum()),'error58_sum':float(b.sum()),'delta_sum':float(delta.sum()),'delta_square_sum':float((delta*delta).sum())}
 assert np.isclose((s['sse415']+s['sse58'])/2,s['common_sse']+s['difference_sse'],atol=1e-6,rtol=1e-12)
 return s

def add(a,b):
 for k,v in b.items():
  if isinstance(v,dict):add(a.setdefault(k,{}),v)
  else:a[k]=a.get(k,0)+v
 return a

def sampled_one(j):
 r,h,Y,P,V=maps(j);e=P-Y
 full={'bands':{b:stat(e[k],V[k]) for k,b in enumerate(('415','58'))},'pair':pair_stat(e,Y,V.all(0))}
 Y=Y[:,yr,xc];P=P[:,yr,xc];V=V[:,yr,xc];e=P-Y
 tr=int(r['tx_row']);tc=int(r['tx_col'])
 E,cross,peak,inc=profile(h,tr,tc,yr,xc,detail=True)
 zone=np.where(E<-.25,0,np.where(E>.25,2,1));near=zone==1
 distance=np.hypot(yr-tr,xc-tc);range_bin=np.digitize(distance,[64,128])
 groups={'all':np.ones(len(E),bool),'clear':zone==0,'critical':zone==1,'blocked':zone==2}
 for width in (.1,.25,.5,1.):groups[f'clearance_abs_le_{width}']=np.abs(E)<=width
 # Label jumps diagnose the target only. They never gate a deployable rule.
 band={}
 for k,b in enumerate(('415','58')):
  i=indices[j];neighbors=np.stack([y[k][i,yr-1,xc],y[k][i,yr+1,xc],y[k][i,yr,xc-1],y[k][i,yr,xc+1]]).astype(np.float64)
  vm=np.unpackbits(packed[k][i],bitorder='big').reshape(256,256).astype(bool)
  good=vm[yr-1,xc]&vm[yr+1,xc]&vm[yr,xc-1]&vm[yr,xc+1]
  jump=(np.max(np.abs(neighbors-Y[k]),axis=0)>5)&good
  g={**groups,'critical_and_label_jump':near&jump,'critical_no_label_jump':near&~jump,'outside_critical_label_jump':~near&jump}
  band[b]={name:stat(e[k],q&V[k]) for name,q in g.items()}
 both=V.all(0);pair={name:pair_stat(e,Y,q&both) for name,q in groups.items()}
 # A partition-based, descriptive variance decomposition. No predictor is fitted or exported.
 partitions={
  'zone':zone,
  'clearance_bins':np.digitize(E,[-1,-.5,-.25,0,.25,.5,1,2,5,10]),
  'zone_and_range':zone*3+range_bin,
  'zone_range_crossings':(zone*3+range_bin)*4+np.minimum(cross//2,3),
  'zone_range_crossings_last_angle':((zone*3+range_bin)*4+np.minimum(cross//2,3))*5+np.digitize(inc,[.1,.25,.5,.75]),
 }
 partition_stats={}
 for name,bins in partitions.items():
  partition_stats[name]={str(int(i)):pair_stat(e,Y,(bins==i)&both) for i in np.unique(bins)}
 rules={}
 for rule in ('clear_equal_shared','clear_high_from_low','clear_low_from_high','all_equal_shared'):
  new=P.copy();gate=np.ones(len(E),bool) if rule.startswith('all') else zone==0
  if 'equal_shared' in rule:
   c=(P[0]+P[1]-gap)/2;new[0,gate]=c[gate];new[1,gate]=c[gate]+gap
  elif rule=='clear_high_from_low':new[1,gate]=P[0,gate]+gap
  else:new[0,gate]=P[1,gate]-gap
  rules[rule]={b:stat(new[k]-Y[k],V[k]) for k,b in enumerate(('415','58'))}
 result={'full':full,'bands':band,'pair':pair,'partitions':partition_stats,'rules':rules}
 return {'j':j,'sample_id':r['sample_id'],'terrain_id':r['terrain_id'],'group':r['group'],'result':result}

def finalize_stat(s):
 if 'sse' in s and 'n' in s:s['rmse_db']=float(np.sqrt(s['sse']/s['n'])) if s['n'] else None
 for v in list(s.values()):
  if isinstance(v,dict):finalize_stat(v)

def sample_summary(total):
 finalize_stat(total)
 for b,g in total['bands'].items():
  a=g['all']
  for name,s in g.items():
   s['pixel_share']=s['n']/a['n'];s['sse_share']=s['sse']/a['sse'];s['rmse_if_errors_in_group_zero']=float(np.sqrt(max(0,a['sse']-s['sse'])/a['n']))
 for name,s in {**total['pair'],'full_pair':total['full']['pair']}.items():
  n=s['n'];energy=s['common_sse']+s['difference_sse']
  s['common_share']=s['common_sse']/energy if energy else None;s['difference_share']=s['difference_sse']/energy if energy else None
  s['common_rmse_db']=float(np.sqrt(s['common_sse']/n)) if n else None;s['difference_rmse_db']=float(np.sqrt(s['difference_sse']/n)) if n else None
  s['actual_delta_mean_db']=gap+s['delta_sum']/n if n else None
  s['delta_std_db']=float(np.sqrt(max(0,s['delta_square_sum']/n-(s['delta_sum']/n)**2))) if n else None
 for name,p in total['partitions'].items():
  n=sum(s['n'] for s in p.values());ds=sum(s['delta_sum'] for s in p.values());d2=sum(s['delta_square_sum'] for s in p.values())
  between=sum(s['delta_sum']**2/s['n'] for s in p.values() if s['n'])-ds*ds/n
  variance=d2-ds*ds/n
  p['_summary']={'n':n,'nonempty_groups':sum(s['n']>0 for s in p.values()),'delta_between_group_variance_share':between/variance,
   'error415_group_mean_sse_share':sum(s['error415_sum']**2/s['n'] for s in p.values() if s['n'])/sum(s['sse415'] for s in p.values()),
   'error58_group_mean_sse_share':sum(s['error58_sum']**2/s['n'] for s in p.values() if s['n'])/sum(s['sse58'] for s in p.values())}
 for rule,bands in total['rules'].items():
  for b,s in bands.items():
   a=total['bands'][b]['all'];s['rmse_change_db']=s['rmse_db']-a['rmse_db'];s['sse_reduction_fraction']=1-s['sse']/a['sse']
 return total

def bootstrap(records,repeats=1000):
 by={}
 for rec in records:
  t=rec['terrain_id'];row=by.setdefault(t,{'group':rec['group'],'values':{}});r=rec['result'];v={}
  for b in ('415','58'):
   for group in ('all','critical'):
    for key in ('n','sse'):v[f'{b}_{group}_{key}']=r['bands'][b][group][key]
   for rule in r['rules']:v[f'{b}_{rule}_sse']=r['rules'][rule][b]['sse']
  v.update({f'full_{k}':r['full']['pair'][k] for k in ('common_sse','difference_sse','n')})
  add(row['values'],v)
 keys=list(next(iter(by.values()))['values']);pos={k:i for i,k in enumerate(keys)};groups={g:np.array([[v['values'][k] for k in keys] for v in by.values() if v['group']==g],float) for g in ('multi','single')}
 rng=np.random.default_rng(2026103103);samples=[]
 for _ in range(repeats):
  sums=sum(a[rng.integers(0,len(a),len(a))].sum(0) for a in groups.values());d={}
  for b in ('415','58'):
   n=sums[pos[f'{b}_all_n']];s=sums[pos[f'{b}_all_sse']]
   d[b+'_critical_sse_share']=sums[pos[f'{b}_critical_sse']]/s
   for rule in records[0]['result']['rules']:d[b+'_'+rule+'_rmse_change_db']=np.sqrt(sums[pos[f'{b}_{rule}_sse']]/n)-np.sqrt(s/n)
  c=sums[pos['full_common_sse']];f=sums[pos['full_difference_sse']];d['full_common_share']=c/(c+f);samples.append(d)
 return {'unit':'terrain, stratified by multi/single; all TX on a terrain kept together','repeats':repeats,'terrain_counts':{g:len(a) for g,a in groups.items()},'intervals_95pct':{k:[float(np.quantile([s[k] for s in samples],q)) for q in (.025,.975)] for k in samples[0]}}

SHIFTS=(-4.,-2.,-1.,-.5,0.,.5,1.,2.,4.)
def dense_one(j):
 r,h,Y,P,V=maps(j);e=P-Y
 yy,xx=np.indices((256,256));rr=yy.ravel();cc=xx.ravel();E=np.empty(len(rr),np.float32)
 for start in range(0,len(rr),4096):E[start:start+4096]=profile(h,int(r['tx_row']),int(r['tx_col']),rr[start:start+4096],cc[start:start+4096])
 E=E.reshape(256,256);blocked=E>0;edge=np.zeros((256,256),bool)
 d=blocked[1:]!=blocked[:-1];edge[1:]|=d;edge[:-1]|=d
 d=blocked[:,1:]!=blocked[:,:-1];edge[:,1:]|=d;edge[:,:-1]|=d
 distance=distance_transform_edt(~edge) if edge.any() else np.full((256,256),np.inf)
 bands={b:{'all':stat(e[k],V[k]),**{f'distance_le_{w}m':stat(e[k],V[k]&(distance<=w)) for w in (1,2,4,8)}} for k,b in enumerate(('415','58'))}
 dy,dx=np.gradient(E);norm=np.hypot(dx,dy);nx=np.divide(dx,norm,out=np.zeros_like(dx),where=norm>1e-6);ny=np.divide(dy,norm,out=np.zeros_like(dy),where=norm>1e-6)
 region=(distance<=4)&(yy>=5)&(yy<251)&(xx>=5)&(xx<251)
 shift_sse={};oracle=e*e
 for shift in SHIFTS:
  new=np.stack([map_coordinates(P[k],np.stack((yy+shift*ny,xx+shift*nx)),order=1,mode='nearest',prefilter=False) for k in (0,1)])
  err=np.where(region[None],new-Y,e);squared=err*err;oracle=np.minimum(oracle,squared)
  shift_sse[str(shift)]={b:float(squared[k,V[k]].sum()) for k,b in enumerate(('415','58'))}
 local={}
 for radius in (1,2,4):
  for k,b in enumerate(('415','58')):
   lo=minimum_filter(P[k],size=2*radius+1,mode='nearest');hi=maximum_filter(P[k],size=2*radius+1,mode='nearest')
   residual=np.maximum(np.maximum(lo-Y[k],Y[k]-hi),0);q=region&V[k]
   full=np.where(region,residual,e[k])
   local.setdefault(str(radius),{})[b]={'all_sse':float((full[V[k]]**2).sum()),'region_n':int(q.sum()),'truth_outside_local_prediction_range_count':int(((Y[k]<lo)|(Y[k]>hi))[q].sum()),'region_baseline_sse':float((e[k,q]**2).sum()),'region_oracle_sse':float((residual[q]**2).sum())}
 result={'bands':bands,'shift_sse':shift_sse,'best_shift_per_pixel_sse':{b:float(oracle[k,V[k]].sum()) for k,b in enumerate(('415','58'))},'local_envelope':local}
 return {'j':j,'terrain_id':r['terrain_id'],'sample_id':r['sample_id'],'group':r['group'],'result':result}

def write(name,value):
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();start=time.time()
 js=list(range(min(8,len(indices)))) if args.smoke else list(range(len(indices)))
 records=[];total={}
 with ThreadPoolExecutor(max_workers=4) as pool:
  for k,r in enumerate(pool.map(sampled_one,js)):
   records.append(r);add(total,json.loads(json.dumps(r['result'])))
   if (k+1)%200==0:print(json.dumps({'phase':'all_maps','maps':k+1,'total':len(js),'seconds':time.time()-start}),flush=True)
 summary=sample_summary(total)
 # Independent reconciliation with the prior same-prediction full-image scorer.
 if not args.smoke:
  previous=json.loads((PRED/'independent_score.json').read_text())['bands']
  for b in ('415','58'):
   assert summary['full']['bands'][b]['n']==previous[b]['valid_pixels']
   assert abs(summary['full']['bands'][b]['rmse_db']-previous[b]['rmse_db'])<1e-10
  summary['terrain_bootstrap']=bootstrap(records)
 summary['scope']={'maps':len(js),'sampled_receivers_per_map':len(yr),'profile_samples':192,'source':'train development only','uses_gpu':False,'model_fitting':False,'frequency_gap_db':gap,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 prefix='smoke_' if args.smoke else ''
 write(prefix+'sampled_and_full.json',summary)
 if not args.smoke:
  with (OUT/'per_map.jsonl').open('w') as f:
   for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
 print(json.dumps({'phase':'sampled_done','seconds':time.time()-start,'full_common_share':summary['full']['pair']['common_share'],'rules':{rule:{b:round(s['rmse_db'],6) for b,s in bands.items()} for rule,bands in summary['rules'].items()}}),flush=True)
 rng=np.random.default_rng(2026103102);dense_js=[]
 for group,count in [('multi',100),('single',28)]:
  candidates=[j for j in range(len(indices)) if rows[indices[j]]['group']==group]
  dense_js.extend(rng.choice(candidates,size=count,replace=False).tolist())
 dense_js=sorted(dense_js[:2] if args.smoke else dense_js);write(prefix+'dense_selection.json',{'seed':2026103102,'prediction_rows':dense_js,'original_rows':[indices[j] for j in dense_js],'selection':'100 multi plus 28 single maps; random before viewing errors'})
 dense_records=[];dense_total={}
 with ThreadPoolExecutor(max_workers=4) as pool:
  for k,r in enumerate(pool.map(dense_one,dense_js)):
   dense_records.append(r);add(dense_total,json.loads(json.dumps(r['result'])))
   if (k+1)%16==0:print(json.dumps({'phase':'dense_boundary','maps':k+1,'total':len(dense_js),'seconds':time.time()-start}),flush=True)
 for b,groups in dense_total['bands'].items():
  a=groups['all']
  for name,s in groups.items():s.update({'rmse_db':float(np.sqrt(s['sse']/s['n'])) if s['n'] else None,'pixel_share':s['n']/a['n'],'sse_share':s['sse']/a['sse'],'rmse_if_group_zero':float(np.sqrt(max(0,a['sse']-s['sse'])/a['n']))})
  best_key=min(dense_total['shift_sse'],key=lambda k:dense_total['shift_sse'][k][b]);best_sse=dense_total['shift_sse'][best_key][b]
  per_map=sum(min(r['result']['shift_sse'][str(s)][b] for s in SHIFTS) for r in dense_records)
  dense_total.setdefault('shift_summary',{})[b]={'best_global_shift_m':float(best_key),'best_global_shift_rmse':float(np.sqrt(best_sse/a['n'])),'best_map_shift_rmse':float(np.sqrt(per_map/a['n'])),'best_pixel_shift_rmse':float(np.sqrt(dense_total['best_shift_per_pixel_sse'][b]/a['n']))}
  for radius,items in dense_total['local_envelope'].items():
   s=items[b];s.update({'all_rmse':float(np.sqrt(s['all_sse']/a['n'])),'overall_sse_reduction_fraction':1-s['all_sse']/a['sse'],'region_sse_reduction_fraction':1-s['region_oracle_sse']/s['region_baseline_sse'],'truth_outside_local_prediction_range_fraction':s['truth_outside_local_prediction_range_count']/s['region_n']})
 dense_total['scope']={'maps':len(dense_js),'terrains':len({r['terrain_id'] for r in dense_records}),'profile_samples':192,'pixel_resolution_m':1,'edge_definition':'both adjacent pixel centers of a 4-neighbor visibility change, excluding exterior map borders','intervention_region':'within 4 m of geometric edge; at least 5 pixels inside map','shift_sign':'positive toward increasing terrain obstruction','oracle_warning':'global/map/pixel best shifts and local envelopes use labels; diagnostics only, not deployable predictions','uses_gpu':False,'seconds_total':time.time()-start}
 write(prefix+'dense_boundary.json',dense_total)
 if not args.smoke:
  with (OUT/'dense_per_map.jsonl').open('w') as f:
   for r in dense_records:f.write(json.dumps(r,separators=(',',':'))+'\n')
 print(json.dumps({'done':True,'smoke':args.smoke,'seconds':time.time()-start,'boundary':dense_total['shift_summary']}),flush=True)
if __name__=='__main__':main()
