import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c20_observation import mesh,interpolation,evaluate_map
from src.c7_surface_features import tx_visibility
O=R/'audit/c20_cell_observation_20261101';S=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
sel=json.loads((O/'selection.json').read_text());pixels=np.load(O/'pixels.npy');cal=np.flatnonzero(np.array(sel['parts'])=='calibration')
pilot=np.random.default_rng(20261118).choice(cal,64,replace=False);qpos=np.arange(16)
(O/'extended_numerics_selection.json').write_text(json.dumps({'positions':pilot.tolist(),'query_positions':qpos.tolist(),'input_only':True},indent=2)+'\n')
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r');meshes=[mesh(d) for d in [0,1]]
rc=np.stack(np.divmod(pixels[pilot,:16],256),-1).astype(float);pre=[]
for _,_,tri in meshes:
 ids,w=interpolation(tri,rc.reshape(-1,2));pre.append((ids.reshape(64,16,3),w.reshape(64,16,3)))
start=time.time()
def one(k):
 j=pilot[k];i=sel['index_rows'][j];row=rows[i];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col'])
 tv=tx_visibility(np.asarray(G[i],np.float32),3-h[tr,tc]%1)[0];res=[]
 for d,(verts,cent,tri) in enumerate(meshes):
  a=evaluate_map(h,tr,tc,rc[k],verts,tri,tv,d,orders=(2,4,8),naz=32,precomputed=(pre[d][0][k],pre[d][1][k]))
  b=evaluate_map(h,tr,tc,rc[k],verts,tri,tv,d,orders=(4,),naz=64,precomputed=(pre[d][0][k],pre[d][1][k]))
  res.append((a,b))
 return res
with ThreadPoolExecutor(max_workers=8) as pool:done=list(pool.map(one,range(64)))
np.save(O/'extended_numerical_values.npy',np.array(done,dtype=object),allow_pickle=True)
def loss(v):return -10*np.log10(np.stack((v[:,0]+v[:,1],v[:,2]+v[:,3]),1).clip(1e-30))
s=np.load(S/'diagnostic_samples.npz');mask=np.load(S/'evaluation_masks.npz')['query_valid'][pilot,:16].reshape(-1,2);zone=s['zone'][pilot,:16].ravel()
checks={}
for d in [0,1]:
 for name in ['full','fixed_diffuse']:
  suf='' if name=='full' else '_fixed_diffuse'
  values={}
  for a,b in [('area4','area16'),('area16','area64')]:
   values[a+'_to_'+b]=np.concatenate([loss(v[d][0][a+suf])-loss(v[d][0][b+suf]) for v in done])
  values['angular32_to64_delta']=np.concatenate([(loss(v[d][0]['area16'+suf])-loss(v[d][0]['point']))-(loss(v[d][1]['area16'+suf])-loss(v[d][1]['point'])) for v in done])
  for mode,diff in values.items():
   for b,band in enumerate(['415','58']):
    for zn,z in [('all',np.ones(len(zone),bool)),('occluded',zone>0),('nominal_blocked',zone==2)]:
     valid=mask[:,b]&z;a=diff[valid,b]
     checks[f'triangle{d}_{name}_{mode}_{band}_{zn}']={'n':int(valid.sum()),'RMS_dB':float(np.sqrt(np.mean(a*a))),'abs_median_dB':float(np.median(abs(a))),'abs_p95_dB':float(np.percentile(abs(a),95)),'fraction_abs_gt1':float(np.mean(abs(a)>1)),'abs_max_dB':float(np.max(abs(a)))}
out={'state':'completed','seconds':time.time()-start,'calibration_terrains':64,'queries_per_terrain':16,'checks':checks,'mask_only_for_reporting_numerical_error_on_scored_points':True,'no_labels_used':True}
(O/'extended_numerics.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'state':'completed','seconds':out['seconds'],'selected':{k:v for k,v in checks.items() if k.startswith('triangle0_full_') and k.endswith('_occluded')}}),flush=True)
