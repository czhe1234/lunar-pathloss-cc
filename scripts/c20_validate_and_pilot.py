import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c20_observation import *
from src.c19_fine_physics import transport
from src.c7_surface_features import tx_visibility
O=R/'audit/c20_cell_observation_20261101';S=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
sel=json.loads((S/'selection.json').read_text());pixels=np.load(S/'pixels.npy');rows=list(csv.DictReader((D/'train_index.csv').open()))
H=np.load(D/'train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r')
meshes=[mesh(d) for d in [0,1]];checks={}
# Geometry exactness for constants, affine fields, and piecewise-linear power fields.
qrc=np.c_[np.arange(16)+100,np.arange(16)+110].astype(float);max_affine=0;max_kernel=0
rng=np.random.default_rng(20261115);vertex_power=rng.uniform(.1,3,(256,256))
for diag,(verts,cent,tri) in enumerate(meshes):
 ids,w=interpolation(tri,qrc)
 a=3+cent[:,0]*.4-cent[:,1]*.1
 max_affine=max(max_affine,float(np.max(abs((a[ids]*w).sum(1)-(3+qrc[:,0]*.4-qrc[:,1]*.1)))))
 pv=vertex_power[verts[:,:,0],verts[:,:,1]]
 avg=pv.mean(1);res=(avg[ids]*w).sum(1)
 rr,cc=qrc.astype(int).T
 kernel=vertex_power[rr,cc]/3+(vertex_power[rr-1,cc]+vertex_power[rr+1,cc]+vertex_power[rr,cc-1]+vertex_power[rr,cc+1])/6
 max_kernel=max(max_kernel,float(np.max(abs(res-kernel))))
 for order in [2,4,8]:
  bary=quadrature(order);np.testing.assert_allclose(bary.mean(0),np.full(3,1/3),atol=1e-14)
assert max_affine<1e-10 and max_kernel<1e-10
checks['affine_interpolation_max_error']=max_affine;checks['piecewise_linear_power_fivepoint_kernel_max_error']=max_kernel
# Continuous query kernel agrees with independently retained C19 integer kernel.
j=17;row=rows[sel['index_rows'][j]];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col']);pp=pixels[j,:32]
rr,cc=np.divmod(pp,256);xyz=np.c_[rr,cc,h[rr,cc]+1]
a=transport(h,tr,tc,pp).reshape(-1,2,16)
errors=[]
for d in [0,1]:
 b=direct(h,tr,tc,xyz,d);errors.append(float(np.max(abs(a[:,d]-b))))
assert max(errors)<1e-10;checks['integer_kernel_agreement']=errors
# Flat LOS gives exact free-space direct power at arbitrary real locations.
h0=np.zeros((256,256));xyz=np.c_[np.linspace(100.17,140.23,24),np.linspace(120.37,161.19,24),np.ones(24)]
d=direct(h0,100,110,xyz,0);assert (d[:,0]==0).all() and np.max(abs(d[:,6:14]))==0
# General continuous rotation swaps diagonal.
row=rows[sel['index_rows'][18]];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col']);rr,cc=np.divmod(pixels[18,:32],256);rr=rr+.17;cc=cc-.21
qh,_,_=surface(h,rr,cc,0);xyz=np.c_[rr,cc,qh+1]
a=direct(h,tr,tc,xyz,0);rot=np.c_[255-cc,rr,qh+1]
b=direct(np.rot90(h),255-tc,tr,rot,1);delta=float(np.max(abs(a-b)));assert delta<1e-7;checks['continuous_rotation_error']=delta
(O/'implementation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
# Input-only numerical pilot; calibration terrains only; no masks or outcomes.
cal=np.flatnonzero(np.array(sel['parts'])=='calibration');pilot=np.random.default_rng(20261116).choice(cal,16,replace=False)
(O/'numerical_pilot_selection.json').write_text(json.dumps({'selection_positions':pilot.tolist(),'query_positions':'first16 fixed C19 uniform queries','uses_labels_or_masks':False},indent=2)+'\n')
start=time.time()
def run(j):
 row=rows[sel['index_rows'][j]];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col']);rc=np.stack(np.divmod(pixels[j,:16],256),1).astype(float)
 tv=tx_visibility(np.asarray(G[sel['index_rows'][j]],np.float32),3-h[tr,tc]%1)[0];out=[]
 for diag,(verts,cent,tri) in enumerate(meshes):
  a=evaluate_map(h,tr,tc,rc,verts,tri,tv,diag,orders=(2,4,8),naz=32)
  b=evaluate_map(h,tr,tc,rc,verts,tri,tv,diag,orders=(4,),naz=64)
  out.append((a,b))
 return out
with ThreadPoolExecutor(max_workers=8) as pool:done=list(pool.map(run,pilot))
def loss(v):return -10*np.log10(np.stack((v[:,0]+v[:,1],v[:,2]+v[:,3]),1).clip(1e-30))
checks2={}
for diag in [0,1]:
 for name,A,B in [('area4_to16','area4','area16'),('area16_to64','area16','area64')]:
  diff=np.concatenate([loss(d[diag][0][A])-loss(d[diag][0][B]) for d in done])
  checks2[f'triangle{diag}_{name}']={'rms_dB':np.sqrt((diff*diff).mean(0)).tolist(),'abs_median_dB':np.median(abs(diff),axis=0).tolist(),'abs_p95_dB':np.percentile(abs(diff),95,axis=0).tolist()}
 diff=np.concatenate([(loss(d[diag][0]['area16'])-loss(d[diag][0]['point']))-(loss(d[diag][1]['area16'])-loss(d[diag][1]['point'])) for d in done])
 checks2[f'triangle{diag}_angular32_to64_delta']={'rms_dB':np.sqrt((diff*diff).mean(0)).tolist(),'abs_median_dB':np.median(abs(diff),axis=0).tolist(),'abs_p95_dB':np.percentile(abs(diff),95,axis=0).tolist()}
np.save(O/'numerical_pilot_values.npy',np.array(done,dtype=object),allow_pickle=True)
result={'state':'completed','seconds':time.time()-start,'calibration_maps':16,'queries_per_map':16,'checks':checks2,'no_labels_or_masks_used':True}
(O/'numerical_pilot.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
