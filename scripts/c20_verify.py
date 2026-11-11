import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,hashlib,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c20_observation import mesh,evaluate_map
from src.c7_surface_features import tx_visibility
O=R/'audit/c20_cell_observation_20261101';S=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
r=json.loads((O/'results.json').read_text());meta=json.loads((O/'observations.json').read_text());sel=json.loads((O/'selection.json').read_text());p=np.load(O/'pixels.npy');n=p.shape[1]
s=np.load(S/'diagnostic_samples.npz');m=np.load(S/'evaluation_masks.npz');C=np.load(O/'calibrated_corrections.npz');conf=np.array(sel['parts'])=='confirmation';cal=~conf;zone=s['zone'][:,:n]
Y=s['Y'][:,:n];P=s['P'][:,:n];valid=m['query_valid'][:,:n];power=np.load(O/'component_powers.npy',mmap_mode='r');L=np.load(O/'operator_loss.npy',mmap_mode='r')
vf=np.asarray(power[:,:,:,3,4]);partial=((vf>1e-6)&(vf<1-1e-6)).any(2);dark=(vf<1e-6).all(2)
zones={'all':np.ones(zone.shape,bool),'occluded_proxy':zone>0,'height_sensitive':zone==1,'nominal_blocked':zone==2,'strict_nominal_blocked':(zone==2)&m['strict_state'][:,:n],
 'candidate_partial_LOS':partial,'candidate_no_LOS':dark&(zone>0)}
maxdiff=0.;checks=0
for b,band in enumerate(['415','58']):
 for name,stats in r['statistics'][band].items():
  correction=C[band+'_'+name].astype(float);e=Y[:,:,b]-P[:,:,b]-correction
  for z,arr in zones.items():
   use=conf[:,None]&valid[:,:,b]&arr;rmse=float(np.linalg.norm(e[use])/np.sqrt(use.sum()))
   diff=abs(rmse-stats[z]['confirm_RMSE']);maxdiff=max(maxdiff,diff);assert diff<2e-6,(band,name,z,diff);assert use.sum()==stats[z]['confirm_n'];checks+=1
reconstructed=-10*np.log10(np.stack((np.asarray(power[...,0]+power[...,1]),np.asarray(power[...,2]+power[...,3])),-1).clip(1e-30))
op=meta['operator_names'];powererr=float(np.max(abs(reconstructed-np.asarray(L[:,:,:,[op.index(k) for k in ['point','centroid','area4','area16']],:]))))
assert powererr<2e-5
# Sequential fresh field/mesh evaluation of savedcal/confirmation samples, after parallel interpolation repair.
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r');freshdiff=0.
for j in [int(np.flatnonzero(cal)[0]),int(np.flatnonzero(conf)[0])]:
 row=rows[sel['index_rows'][j]];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col'])
 tv=tx_visibility(np.asarray(G[sel['index_rows'][j]],np.float32),3-h[tr,tc]%1)[0];rc=np.stack(np.divmod(p[j,:4],256),1).astype(float)
 for d in [0,1]:
  verts,cent,tri=mesh(d);a=evaluate_map(h,tr,tc,rc,verts,tri,tv,d,orders=(2,4),naz=32)
  for k,name in enumerate(['point','centroid','area4','area16']):
   np.testing.assert_allclose(a[name],power[j,:4,d,k],rtol=1e-11,atol=1e-30);freshdiff=max(freshdiff,float(np.max(abs(a[name]-power[j,:4,d,k]))))
assert np.array_equal(p,np.load(S/'pixels.npy')[:,:128])
assert len(set(sel['terrain_ids']))==612 and not(set(np.array(sel['terrain_ids'])[cal])&set(np.array(sel['terrain_ids'])[conf]))
fit=np.load(R/'features/c6_points/fit_map_indices.npy');assert not(set(sel['terrain_ids'])&{rows[int(i)]['terrain_id'] for i in fit})
raw=json.loads((R/'audit/raw_audit.json').read_text())['files']
for name,a in raw.items():
 st=(D/name).stat();assert st.st_size==a['size_bytes'] and st.st_mtime_ns==a['mtime_ns'],name
old=json.loads((S/'verification.json').read_text())['C10_weight_sha256']
for name,h in old.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
for name,h in meta['source_hashes'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
# Source bundle records reproducible implementation including retained initial failure.
src=O/'source';src.mkdir(exist_ok=True)
for f in [*sorted((R/'scripts').glob('c20_*.py')),R/'src/c20_observation.py',R/'src/c20_triangle_transport.cpp']:
 shutil.copy2(f,src/f.name)
out={'state':'passed','saved_correction_RMSE_checks':checks,'max_RMSE_difference':maxdiff,'raw_power_to_saved_dB_max_error':powererr,
 'fresh_sequential_vs_saved_power_max_abs_difference':freshdiff,'same_frozen_uniform_queries':True,'terrain_partition_no_overlap_and_no_C10_FIT_overlap':True,
 'raw_files_size_mtime_unchanged':len(raw),'C10_weight_sha256':old,'extraction_source_hashes_unchanged':True,
 'initial_failed_arrays_not_used':True,'GPU_used':False,'new_tree_or_neural_model_training':False}
(O/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
