"""Combine complementary input descriptors and express critical TX clearance
relative to a family of raster-height phases. These are hypotheses, not actual TX.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,numpy as np,time
R=Path(__file__).resolve().parents[1];O=R/'features/c9_points';O.mkdir(exist_ok=True)
C=R/'features/c8_points';P=R/'features/c6_points';S=R/'features/c7_surface_points'
cm=json.loads((C/'metadata.json').read_text());pm=json.loads((P/'metadata.json').read_text());sm=json.loads((S/'metadata.json').read_text())
phases=np.arange(8)/8
names=cm['features']+pm['features'][31:]+sm['features'][31:]+[f'critical_TX_floorphase_{k}' for k in range(8)]
start=time.time()
for part in ['fit','dev']:
 cx=np.load(C/f'{part}_x.npy',mmap_mode='r');px=np.load(P/f'{part}_x.npy',mmap_mode='r');sx=np.load(S/f'{part}_x.npy',mmap_mode='r')
 assert not (O/f'{part}_complete.json').exists()
 out=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float32,shape=(len(cx),len(names)))
 for lo in range(0,len(cx),16384):
  sl=slice(lo,min(lo+16384,len(cx)));v=cx[sl]
  # Critical total TX height above the raster ground = 2.5 + 2*sinh(3*x27).
  # Compare to 3-frac(H_TX+phase), solely an explicit input transform.
  critical=2.5+2*np.sinh(3*v[:,27]);txh=v[:,31]
  extra=np.arcsinh((critical[:,None]-(3-np.remainder(txh[:,None]+phases,1)))/2)/3
  out[sl]=np.concatenate((v,px[sl,31:],sx[sl,31:],extra),1)
 out.flush();(O/f'{part}_complete.json').write_text(json.dumps({'complete':True,'points':len(cx),'seconds':time.time()-start})+'\n')
meta={**cm,'features':names,'C9_inputs':'C8 TX context + ordered profile + weighted surrounding surface geometry + input-only phase-conditioned critical clearance','no_target_or_mask_read_by_builder':True,'phase_is_hypothesis_not_recovered_TX':True}
(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'complete':True,'features':len(names),'seconds':time.time()-start}),flush=True)

