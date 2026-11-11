"""Same-density confirmation of the height sensitivity, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
s=importlib.util.spec_from_file_location('alignment',Path(__file__).with_name('diagnose_geometry_alignment.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a);z=np.load(a.OUT/'development_geometry_samples.npz');e2=np.load(a.OUT/'development_E2.npy');eold=z['E'];start=time.time()
def one(i):
 r=a.rows[int(i)];h=a.H[int(r['hm_row'])];return a.profile(h,int(r['tx_row']),int(r['tx_col']),a.rr,a.cc,n=384)[0]
with ThreadPoolExecutor(max_workers=4) as pool:e3=np.stack(list(pool.map(one,z['original_rows'])))
np.save(a.OUT/'development_E3_384.npy',e3);v=z['valid'];error=z['predictions'].astype(np.float64)-z['labels'];result={}
for k,b in enumerate(['415','58']):
 totaln=int(v[:,k].sum());ee=error[:,k];totalsse=float((ee[v[:,k]]**2).sum());q=(e3<-.25)&(e2>=-.25)&v[:,k];changed=(np.digitize(e3,[-.25,.25])!=np.digitize(eold,[-.25,.25]))&v[:,k];result[b]={'same_density_sensitivity_pixels':int(q.sum()),'same_density_sensitivity_pixel_share':float(q.sum())/totaln,'same_density_sensitivity_sse_share':float((ee[q]**2).sum())/totalsse,'density_only_changed_state_pixel_share':float(changed.sum())/totaln}
a.write('geometry_sampling_match.json',{'scope':'all2562devmaps same384point profiles for TX3 andTX2; no model inference or training','results':result,'max_abs_E3_density_change':float(abs(e3-eold).max()),'seconds':time.time()-start});print(json.dumps({'done':True,'results':result,'seconds':time.time()-start}),flush=True)
