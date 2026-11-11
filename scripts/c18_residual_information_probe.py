"""Fixed out-of-base-training residual probes; CPU only, diagnostic not submission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,time,hashlib,shutil
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[1];D=R/'features/c6_points';O=R/'audit/c18_identifiability_20261101/residual_probe';O.mkdir(exist_ok=False);(O/'source').mkdir()
for p in [Path(__file__),R/'audit/c18_identifiability_20261101/protocol.json']:shutil.copy2(p,O/'source'/p.name)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
ids=np.load(D/'dev_map_indices.npy');X=np.array(np.load(R/'features/c10_points/dev_x.npy',mmap_mode='r'),dtype=np.float32)
extra=np.load(R/'features/c14_receiver_points/dev_x.npy',mmap_mode='r')
Y=np.array(np.load(D/'dev_y.npy',mmap_mode='r'),float);M=np.array(np.load(D/'dev_mask.npy',mmap_mode='r'),bool)
P=np.column_stack([np.load(R/f'runs/c10_points_phase_geometry_s20261101/dev_pred{b}.npy') for b in ['415','58']]).astype(float)
assert len(X)==len(ids)*256 and X.shape[1]==294 and extra.shape==(len(X),103)
assert np.isfinite(X).all() and np.isfinite(extra).all()
mapterr=np.array([rows[int(i)]['terrain_id'] for i in ids]);terr=sorted(set(mapterr));assert len(terr)==612
rng=np.random.default_rng(20261102);rng.shuffle(terr);calterr=set(terr[:306]);confterr=terr[306:]
cal=np.repeat(np.isin(mapterr,list(calterr)),256);conf=~cal
pointterr=np.repeat(mapterr,256)
assert set(pointterr[cal]).isdisjoint(pointterr[conf])
confcode=np.array([{t:i for i,t in enumerate(confterr)}[t] for t in pointterr[conf]])
e3=(2*np.sinh(3*X[:,11:19])).max(1);e2=2*np.sinh(3*X[:,19])
zones={'all':np.ones(len(X),bool),'occluded_proxy':e2>=0,'height_sensitive':(e3<0)&(e2>=0),'nominal_blocked':e3>=0,'stable_clear':e2<-.5}
names=json.loads((R/'features/c10_points/metadata.json').read_text())['features']
extra_names=json.loads((R/'features/c14_receiver_points/metadata.json').read_text())['features']
params={'objective':'regression','metric':'l2','num_leaves':31,'max_depth':8,'min_data_in_leaf':200,'max_bin':255,'learning_rate':.05,'lambda_l2':1.,'feature_fraction':1.,'bagging_fraction':1.,'bagging_freq':0,'verbosity':-1,'num_threads':8,'seed':20261102,'deterministic':True,'force_col_wise':True,'device_type':'cpu'}
manifest={'calibration_terrains':sorted(calterr),'confirmation_terrains':sorted(confterr),
 'calibration_maps':int(cal.sum()//256),'confirmation_maps':int(conf.sum()//256),
 'calibration_points':int(cal.sum()),'confirmation_points':int(conf.sum()),
 'parameters':params,'trees':400,'no_early_stopping':True,
 'occluded_proxy_definition':'nominal TX2m maximum clearance>=0, union of input-defined height-sensitive and nominal-blocked points',
 'limitation':'These DEV terrains did not enter C10 weight fitting, but prior model selection/diagnosis has used this DEV set. This is a diagnostic confirmation split, not a new pristine benchmark.',
 'C10_FIT_weights_sha256':{b:hashlib.sha256((R/f'runs/c10_points_phase_geometry_s20261101/band{b}.txt').read_bytes()).hexdigest() for b in ['415','58']}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
np.save(O/'confirmation_point_indices.npy',np.flatnonzero(conf))
predictions={'C10':P.copy(),'constant':P.copy()};res=Y-P
for k in [0,1]:predictions['constant'][:,k]+=res[cal&M[:,k],k].mean()
start=time.time();fitstats={}
for name in ['same_C10_inputs','C10_plus_receiver_geometry','shuffled_calibration_residual_control']:
 A=np.concatenate((X,np.array(extra)),1) if name=='C10_plus_receiver_geometry' else X
 featnames=names+extra_names if A.shape[1]==397 else names
 new=P.copy();fitstats[name]={}
 for k,b in enumerate(['415','58']):
  keep=cal&M[:,k];target=res[keep,k].copy()
  if name.startswith('shuffled'):target=np.random.default_rng(20261102+k).permutation(target)
  ds=lgb.Dataset(A[keep],label=target,feature_name=featnames)
  model=lgb.train(params,ds,num_boost_round=400)
  correction=model.predict(A,num_threads=8);new[:,k]+=correction
  model.save_model(str(O/f'{name}_{b}.txt'))
  gain=model.feature_importance(importance_type='gain');top=np.argsort(gain)[-12:][::-1]
  fitstats[name][b]={'calibration_valid_pixels':int(keep.sum()),'calibration_RMSE':float(np.sqrt(np.mean((new[keep,k]-Y[keep,k])**2))),'top_features':[{'name':featnames[i],'gain':float(gain[i])} for i in top]}
  print(json.dumps({'probe':name,'band':b,'seconds':time.time()-start,'confirmation_RMSE':float(np.sqrt(np.mean((new[conf&M[:,k],k]-Y[conf&M[:,k],k])**2)))}),flush=True)
  del ds,model
 predictions[name]=new
 if A is not X:del A
joint=M.all(1);privileged=P.copy();privcoef=[]
for k in [0,1]:
 m=cal&joint;a=res[m,1-k];y=res[m,k];slope=float(np.sum((a-a.mean())*(y-y.mean()))/np.sum((a-a.mean())**2));intercept=float(y.mean()-slope*a.mean())
 privileged[:,k]+=intercept+slope*res[:,1-k];privcoef.append({'intercept':intercept,'slope':slope})
predictions['privileged_other_band_observation']=privileged
bootrng=np.random.default_rng(20261103);weights=bootrng.multinomial(306,np.full(306,1/306),size=1000)
stats={};sse_by={};count_by={}
for pname,pr in predictions.items():
 stats[pname]={};sse_by[pname]={};count_by[pname]={}
 np.save(O/f'{pname}_confirmation_prediction.npy',pr[conf].astype(np.float32))
 for zn,z in zones.items():
  stats[pname][zn]={};sse_by[pname][zn]={};count_by[pname][zn]={}
  for k,b in enumerate(['415','58']):
   for scope in ['band_valid','joint_valid']:
    v=(z&M[:,k]&(joint if scope=='joint_valid' else True))[conf]
    if pname=='privileged_other_band_observation' and scope=='band_valid':continue
    err=(pr[conf,k]-Y[conf,k]);ss=np.bincount(confcode,weights=err*err*v,minlength=306);cc=np.bincount(confcode,weights=v.astype(float),minlength=306)
    key=f'{b}_{scope}';sse_by[pname][zn][key]=ss;count_by[pname][zn][key]=cc
    base_ss=sse_by['C10'][zn][key] if pname!='C10' else ss
    bootbase=weights@base_ss;bootnew=weights@ss
    reduction=100*(1-ss.sum()/base_ss.sum())
    interval=np.percentile(100*(1-bootnew/np.maximum(bootbase,1e-15)),[2.5,97.5]).tolist()
    stats[pname][zn][key]={'valid_pixels':int(cc.sum()),'RMSE':float(np.sqrt(ss.sum()/cc.sum())),'SSE_reduction_percent_vs_C10':float(reduction),'terrain_bootstrap95_SSE_reduction_percent':interval}
comparisons={}
for zn in zones:
 comparisons[zn]={}
 for k,b in enumerate(['415','58']):
  key=f'{b}_band_valid';a=sse_by['same_C10_inputs'][zn][key];new=sse_by['C10_plus_receiver_geometry'][zn][key]
  comparisons[zn][b]={'additional_SSE_reduction_percent':float(100*(1-new.sum()/a.sum())),'terrain_bootstrap95':np.percentile(100*(1-(weights@new)/np.maximum(weights@a,1e-15)),[2.5,97.5]).tolist()}
decision={}
for b in ['415','58']:
 key=f'{b}_band_valid';v=stats['same_C10_inputs']['occluded_proxy'][key];neg=stats['shuffled_calibration_residual_control']['occluded_proxy'][key];extra_cmp=comparisons['occluded_proxy'][b]
 decision[b]={'existing_information_predictability_supported':bool(v['SSE_reduction_percent_vs_C10']>=5 and v['terrain_bootstrap95_SSE_reduction_percent'][0]>0 and v['RMSE']<neg['RMSE']),
 'added_geometry_increment_supported':bool(extra_cmp['additional_SSE_reduction_percent']>=5 and extra_cmp['terrain_bootstrap95'][0]>0)}
out={'state':'completed','scope':'Out-of-C10-fitting residual calibration and terrain-disjoint confirmation; no official VAL score',
 'statistics':stats,'extra_geometry_vs_same_input':comparisons,'fit_statistics':fitstats,
 'privileged_other_band_coefficients':privcoef,'decision':decision,'seconds':time.time()-start,
 'bootstrap':'1000 paired resamples of306 confirmation terrains, retaining all sample maps/pixels per selected terrain',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(O/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'state':'completed','decision':decision,'seconds':out['seconds'],'occluded':{n:stats[n]['occluded_proxy'] for n in stats}}),flush=True)
