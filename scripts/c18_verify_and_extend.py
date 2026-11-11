"""Recompute saved prediction scores independently and add fixed terrain-bias control."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/c18_identifiability_20261101';D=R/'features/c6_points'
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));ids=np.load(D/'dev_map_indices.npy')
X=np.load(R/'features/c10_points/dev_x.npy',mmap_mode='r');Y=np.load(D/'dev_y.npy',mmap_mode='r').astype(float);M=np.load(D/'dev_mask.npy',mmap_mode='r').astype(bool)
P=np.column_stack([np.load(R/f'runs/c10_points_phase_geometry_s20261101/dev_pred{b}.npy') for b in ['415','58']]).astype(float)
checks=[];maxdiff=0.
main=O/'residual_probe';man=json.loads((main/'manifest.json').read_text());idx=np.load(main/'confirmation_point_indices.npy')
cal=set(man['calibration_terrains']);conf=set(man['confirmation_terrains']);officialdev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
assert len(cal)==len(conf)==306 and cal.isdisjoint(conf) and cal|conf==officialdev
checks.append('Main calibration/confirmation terrain split is disjoint and covers exactly612 internalDEV terrains')
def zones(indices):
 v=np.array(X[indices]);e3=(2*np.sinh(3*v[:,11:19])).max(1);e2=2*np.sinh(3*v[:,19])
 return {'all':np.ones(len(indices),bool),'occluded_proxy':e2>=0,'height_sensitive':(e3<0)&(e2>=0),'nominal_blocked':e3>=0,'stable_clear':e2<-.5}
z=zones(idx);result=json.loads((main/'result.json').read_text());recalc={}
for name,stats in result['statistics'].items():
 pr=np.load(main/f'{name}_confirmation_prediction.npy').astype(float);recalc[name]={}
 for zn in z:
  for k,b in enumerate(['415','58']):
   scope='joint_valid' if name.startswith('privileged') else 'band_valid';key=f'{b}_{scope}';m=z[zn]&M[idx,k]
   if scope=='joint_valid':m&=M[idx].all(1)
   value=float(np.sqrt(np.mean((pr[m,k]-Y[idx,k][m])**2)));ref=stats[zn][key]
   assert m.sum()==ref['valid_pixels'];diff=abs(value-ref['RMSE']);assert diff<1e-5,(name,zn,b,diff);maxdiff=max(maxdiff,diff)
   if zn=='occluded_proxy':recalc[name][b]=value
checks.append('Main probe saved FP32 predictions independently reproduce all zone RMSE/counts within1e-5dB')
C=O/'terrain_crossover';manifest=json.loads((C/'manifest.json').read_text());qi=np.load(C/'query_point_indices.npy');qx=zones(qi)
rmap={int(i):j for j,i in enumerate(ids)};assign=manifest['terrain_assignments'];qmap=manifest['query_index_rows']
qt=np.repeat([rows[int(i)]['terrain_id'] for i in qmap],256);dist=[]
# Reconstruct the exact input-only support selection, including equalization.
byter={}
for j,i in enumerate(ids):
 row=rows[int(i)]
 if row['group']=='multi':byter.setdefault(row['terrain_id'],{}).setdefault((int(row['tx_row']),int(row['tx_col'])),j)
terr=sorted(byter);rng=np.random.default_rng(20261104);rng.shuffle(terr);sup=[[],[]];q=[];fold={t:int(j>=20) for j,t in enumerate(terr)}
for t in terr:
 maps=sorted(byter[t].values());rng.shuffle(maps);cut=len(maps)//2;s=maps[:cut];qq=maps[cut:]
 assert ids[s].tolist()==assign[t]['support_index_rows'] and ids[qq].tolist()==assign[t]['query_index_rows']
 sup[fold[t]].extend(s);q.extend(qq)
point=lambda v:(np.asarray(v)[:,None]*256+np.arange(256)[None]).ravel()
assert np.array_equal(qi,point(q))
si=[point(v) for v in sup];n=min(map(len,si))
for f in [0,1]:
 if len(si[f])>n:si[f]=np.sort(rng.choice(si[f],n,replace=False))
 np.save(C/f'calibration_point_indices_fold{f}.npy',si[f])
for t,v in assign.items():
 s={(rows[i]['tx_row'],rows[i]['tx_col']) for i in v['support_index_rows']}
 q={(rows[i]['tx_row'],rows[i]['tx_col']) for i in v['query_index_rows']}
 assert s.isdisjoint(q)
checks.append('Cross-over support/query TX coordinates are disjoint; identical inputs were deduplicated')
pt=np.repeat([rows[int(i)]['terrain_id'] for i in ids],256);mean_pred=P[qi].copy();terrain_bias={}
for t in terr:
 f=fold[t];s=si[f][pt[si[f]]==t];tgt=qt==t;terrain_bias[t]=[]
 for k in [0,1]:
  v=s[M[s,k]];mu=float(np.mean(Y[v,k]-P[v,k]));terrain_bias[t].append(mu);mean_pred[tgt,k]+=mu
np.save(C/'same_terrain_constant_query_prediction.npy',mean_pred.astype(np.float32))
for i in qmap:
 row=rows[i];coord=np.array([int(row['tx_row']),int(row['tx_col'])]);support=np.array([[int(rows[j]['tx_row']),int(rows[j]['tx_col'])] for j in assign[row['terrain_id']]['support_index_rows']])
 dist.append(float(np.sqrt(((support-coord)**2).sum(1)).min()))
dd=np.repeat(dist,256);code={t:j for j,t in enumerate(terr)};tc=np.array([code[t] for t in qt])
rng=np.random.default_rng(20261105);weights=rng.multinomial(39,np.full(39,1/39),size=1000)
preds={name:np.load(C/f'{name}_query_prediction.npy').astype(float) for name in ['C10','same_terrain_other_TX_support','different_terrain_support']};preds['same_terrain_constant']=mean_pred
cres=json.loads((C/'result.json').read_text());extensions={}
for name,pr in preds.items():
 extensions[name]={}
 for zn in ['all','occluded_proxy','height_sensitive','nominal_blocked']:
  extensions[name][zn]={}
  for k,b in enumerate(['415','58']):
   m=qx[zn]&M[qi,k];err=pr[:,k]-Y[qi,k];ss=np.bincount(tc,weights=err*err*m,minlength=39);base=np.bincount(tc,weights=(P[qi,k]-Y[qi,k])**2*m,minlength=39)
   rmse=float(np.sqrt(ss.sum()/m.sum()));extensions[name][zn][b]={'valid_pixels':int(m.sum()),'RMSE':rmse,'SSE_reduction_percent':float(100*(1-ss.sum()/base.sum())),'terrain_bootstrap95':np.percentile(100*(1-(weights@ss)/(weights@base)),[2.5,97.5]).tolist()}
   if name in cres['statistics']:
    ref=cres['statistics'][name][zn][b];assert int(m.sum())==ref['valid_pixels'];diff=abs(rmse-ref['RMSE']);assert diff<1e-5;maxdiff=max(maxdiff,diff)
checks.append('Cross-over saved predictions independently reproduce all reported scores/counts within1e-5dB')
distance={}
for lo,hi in [(0,8),(8,16),(16,32),(32,float('inf'))]:
 name=f'{lo}_{hi}';distance[name]={}
 for k,b in enumerate(['415','58']):
  m=(dd>=lo)&(dd<hi)&qx['occluded_proxy']&M[qi,k];baseline=((P[qi,k]-Y[qi,k])**2)[m].sum()
  distance[name][b]={'valid_pixels':int(m.sum()),'RMSE':{pname:float(np.sqrt(np.mean((pr[m,k]-Y[qi,k][m])**2))) if m.any() else None for pname,pr in preds.items()},
  'SSE_reduction_percent':{pname:float(100*(1-((pr[m,k]-Y[qi,k][m])**2).sum()/baseline)) if baseline else None for pname,pr in preds.items()}}
(C/'fixed_controls.json').write_text(json.dumps({'terrain_constant_correction':terrain_bias,'statistics':extensions,'nearest_support_TX_distance':distance,'scope':'Diagnostic support-label controls, not deployable predictions. Distance analysis exploratory; all predefined bins retained.'},indent=2)+'\n')
raw=json.loads((R/'audit/raw_audit.json').read_text())
for name,ref in raw['files'].items():
 st=(R.parent/'data'/name).stat();assert st.st_size==ref['size_bytes'] and st.st_mtime_ns==ref['mtime_ns']
for b,sha in man['C10_FIT_weights_sha256'].items():
 assert hashlib.sha256((R/f'runs/c10_points_phase_geometry_s20261101/band{b}.txt').read_bytes()).hexdigest()==sha
checks.append('All18 raw files retain audited size/mtime; frozen C10 FIT weights retain exact SHA256')
source=O/'source';source.mkdir(exist_ok=True)
for p in (R/'scripts').glob('c18_*.py'):shutil.copy2(p,source/p.name)
for p in O.glob('*protocol.json'):shutil.copy2(p,source/p.name)
out={'checks':checks,'max_saved_prediction_RMSE_difference_db':maxdiff,'main_occluded_RMSE_recomputed':recalc,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}}
(O/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
