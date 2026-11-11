import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,time,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c20_observation import mesh,evaluate_map
from src.c7_surface_features import tx_visibility
O=R/'audit/c20_cell_observation_20261101';S=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
sel=json.loads((S/'selection.json').read_text());pixels=np.load(S/'pixels.npy')[:,:128]
prot=json.loads((O/'protocol.json').read_text());prot['primary_queries']='First128 of originally frozen256 uniform queries, all612 terrains. Reduced before any outcomes/operator confirmation examined, to limit CPU cost. All paired scores use identical128queries.'
(O/'protocol.json').write_text(json.dumps(prot,indent=2)+'\n')
(O/'selection.json').write_text(json.dumps({**sel,'pixels_per_map':128,'selection_rule':'first128 of fixedC19 uniform pixels'},indent=2)+'\n');np.save(O/'pixels.npy',pixels)
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r')
meshes=[mesh(d) for d in [0,1]]
from src.c20_observation import interpolation
allrc=np.stack(np.divmod(pixels,256),-1).reshape(-1,2).astype(float)
precalc=[interpolation(t,allrc) for _,_,t in meshes]
precalc=[(ids.reshape(612,128,3),w.reshape(612,128,3)) for ids,w in precalc]
print("serial interpolation indices and weights ready",flush=True)
Pdir=R/'runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh'
pos={i:j for j,i in enumerate(json.loads((Pdir/'indices.json').read_text())['index_rows'])}
P=[np.load(Pdir/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ops=['point','centroid','area4','area16','centroid_dbinterp','area4_dbinterp','area16_dbinterp','centroid_fixed_diffuse','area4_fixed_diffuse','area16_fixed_diffuse']
loss=np.lib.format.open_memmap(O/'operator_loss.npy',mode='w+',dtype='float32',shape=(612,128,2,len(ops),2))
raw=np.lib.format.open_memmap(O/'component_powers.npy',mode='w+',dtype='float64',shape=(612,128,2,4,5))
post=np.lib.format.open_memmap(O/'C10_fivepoint_controls.npy',mode='w+',dtype='float32',shape=(612,128,2,2))
start=time.time()
def one(j):
 i=sel['index_rows'][j];row=rows[i];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col'])
 rr,cc=np.divmod(pixels[j],256);rc=np.stack((rr,cc),1).astype(float)
 tv=tx_visibility(np.asarray(G[i],np.float32),3-h[tr,tc]%1)[0]
 for d,(verts,cent,tri) in enumerate(meshes):
  a=evaluate_map(h,tr,tc,rc,verts,tri,tv,d,orders=(2,4),naz=32,precomputed=(precalc[d][0][j],precalc[d][1][j]))
  for k,n in enumerate(ops):
   if n.endswith('dbinterp'):ll=a[n]
   else:ll=-10*np.log10(np.stack((a[n][:,0]+a[n][:,1],a[n][:,2]+a[n][:,3]),1).clip(1e-30))
   loss[j,:,d,k]=ll
  for k,n in enumerate(['point','centroid','area4','area16']):raw[j,:,d,k]=a[n]
 for b in [0,1]:
  pp=np.asarray(P[b][pos[i]],float);ps=np.stack((pp[rr,cc],pp[rr-1,cc],pp[rr+1,cc],pp[rr,cc-1],pp[rr,cc+1]),1);w=np.array([1/3,1/6,1/6,1/6,1/6])
  post[j,:,0,b]=ps@w;post[j,:,1,b]=-10*np.log10(np.power(10.,-ps/10)@w)
 return j
with ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(one,j) for j in range(612)]
 for n,f in enumerate(as_completed(futures),1):
  f.result()
  if n%16==0:print(json.dumps({'completed':n,'total':612,'seconds':time.time()-start}),flush=True)
for a in [loss,raw,post]:a.flush()
meta={'state':'completed','seconds':time.time()-start,'CPU_workers':8,'query_pixels_per_map':128,
 'operator_names':ops,'operator_loss_shape':list(loss.shape),'component_power_names':['direct415','diffuse415','direct58','diffuse58','visible_fraction'],'component_operator_names':['point','centroid','area4','area16'],
 'C10_control_names':['fivepoint_dB','fivepoint_linear_power'],'source_hashes':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'src/c20_observation.py',R/'src/c20_triangle_transport.cpp']},
 'point_and_cell_surrogate_fields_do_not_use_labels_masks_or_C10':True}
(O/'observations.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
