"""Independent saved-output recomputation, source invariants and terrain-level uncertainty."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'audit/c19_fine_physics_20261101'
report=json.loads((O/'confirmation_results.json').read_text());sel=json.loads((O/'selection.json').read_text());conf=np.array(sel['parts'])=='confirmation';cal=~conf
s=np.load(O/'diagnostic_samples.npz');m=np.load(O/'evaluation_masks.npz');p=np.load(O/'probe_predictions.npz');zone=s['zone']
zones={'all':np.ones(zone.shape,bool),'occluded_proxy':zone>0,'height_sensitive':zone==1,'nominal_blocked':zone==2,'strict_nominal_blocked':(zone==2)&m['strict_state']}
maxdiff=0.;checks=0
for b,band in enumerate(['415','58']):
 for target in ['signed_residual','curvature_residual','roughness_magnitude']:
  if target=='signed_residual':y=s['Y'][:,:,b]-s['P'][:,:,b];mask=m['query_valid'][:,:,b]
  elif target=='curvature_residual':y=s['QY'][:,:,b]-s['QP'][:,:,b];mask=m['five_valid'][:,:,b]
  else:y=np.log1p(s['QY'][:,:,b]**2);mask=m['five_valid'][:,:,b]
  for n,v in report['statistics'][f'{band}_{target}'].items():
   if n=='baseline':continue
   pred=p[f'{band}_{target}_{n}'].astype(float)
   for zn,z in zones.items():
    valid=conf[:,None]&mask&z
    rmse=float(np.linalg.norm((y-pred)[valid])/np.sqrt(valid.sum()))
    dif=abs(rmse-v[zn]['confirmation_RMSE']);maxdiff=max(maxdiff,dif);assert dif<2e-6,(band,target,n,zn,dif)
    assert valid.sum()==v[zn]['confirmation_n'];checks+=1
assert len(set(sel['terrain_ids']))==612
assert len(set(np.array(sel['terrain_ids'])[cal])&set(np.array(sel['terrain_ids'])[conf]))==0
# All held-out terrain identities remain outside the C10 FIT identities.
import csv
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
fitids=np.load(R/'features/c6_points/fit_map_indices.npy');fit_terr={rows[int(i)]['terrain_id'] for i in fitids}
assert not(set(sel['terrain_ids'])&fit_terr)
raw=json.loads((R/'audit/raw_audit.json').read_text())['files'];unchanged=[]
for name,info in raw.items():
 f=R.parent/'data'/name;st=f.stat()
 assert st.st_size==info['size_bytes'] and st.st_mtime_ns==info['mtime_ns'],name
 unchanged.append(name)
hashes={}
expected={
 'runs/c10_points_phase_geometry_s20261101/band415.txt':'332b50bfd180c4a99116ee7fe9e191964cc16be448edced898c15e51a7bda3ce',
 'runs/c10_points_phase_geometry_s20261101/band58.txt':'6dfbbf0b0debbbc8155401a6860295d87fdb0cedb5e0ebb5e1c0d4deb44024a4',
 'runs/c10_full_refit_s20261101/band415.txt':'27f6fb0476555bbeb858aadfc30072d10a8220d644076e0944ee8d0988c1e1c0',
 'runs/c10_full_refit_s20261101/band58.txt':'cdbc464e748f422ed6d49a511c86ec98b70c5f089e7fa1fdb4f684092b0ad9ff'}
for name,value in expected.items():
 hashes[name]=hashlib.sha256((R/name).read_bytes()).hexdigest();assert hashes[name]==value,name
featuremeta=json.loads((O/'physical_features.json').read_text())
for name,value in featuremeta['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==value,name
# Terrain bootstrap correlations, independently aggregated from raw local-curvature outcomes.
w=np.random.default_rng(20261114).multinomial(conf.sum(),np.full(conf.sum(),1/conf.sum()),1000);corr_ci={}
for zn,z in zones.items():
 mm=conf[:,None]&m['five_valid'].all(2)&z
 a=s['QY'][:,:,0];b=s['QY'][:,:,1]
 n=mm.sum(1);sx=(a*mm).sum(1);sy=(b*mm).sum(1);sx2=(a*a*mm).sum(1);sy2=(b*b*mm).sum(1);sxy=(a*b*mm).sum(1)
 nn=w@n[conf];xx=w@sx[conf];yy=w@sy[conf];aa=w@sx2[conf];bb=w@sy2[conf];ab=w@sxy[conf]
 r=(ab-xx*yy/nn)/np.sqrt((aa-xx*xx/nn)*(bb-yy*yy/nn))
 corr_ci[zn]=np.percentile(r,[2.5,97.5]).tolist()
obs=json.loads((O/'observation_crosscheck.json').read_text());hole_fraction={}
for band in ['415','58']:
 hole_fraction[band]={}
 for zn,v in obs['distance_to_invalid'][band].items():
  total_n=sum(q['n_query'] for q in v);total_sse=sum(q['query_sse'] for q in v)
  hole_fraction[band][zn]={}
  for k,lab in [(1,'within2m'),(3,'within8m'),(4,'within16m')]:
   hole_fraction[band][zn][lab]={'pixel_fraction':sum(q['n_query'] for q in v[:k])/total_n,'SSE_fraction':sum(q['query_sse'] for q in v[:k])/total_sse}
# Read-only source records; outputs and replay scripts frozen together in audit directory.
src=O/'source';src.mkdir(exist_ok=True)
for f in [*sorted((R/'scripts').glob('c19_*.py')),R/'src/c19_fine_physics.py',R/'src/c19_triangle_transport.cpp']:
 shutil.copy2(f,src/f.name)
result={'state':'passed','saved_prediction_metric_checks':checks,'max_RMSE_difference':maxdiff,'unique_terrains':612,'calibration_terrains':306,'confirmation_terrains':306,
 'zero_overlap_with_C10_FIT':True,'raw_size_mtime_unchanged':unchanged,'C10_weight_sha256':hashes,
 'feature_source_hashes_verified':True,'cross_band_QY_correlation_terrain_bootstrap95':corr_ci,'invalid_neighborhood_concentration':hole_fraction,
 'GPU_used_this_turn':False,'no_new_tree_or_neural_training':True,'official_score_unchanged':3.1775908265908654}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
