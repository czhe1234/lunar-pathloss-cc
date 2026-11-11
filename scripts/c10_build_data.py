"""C10: input-only path descriptors under the phase-zero TX-height hypothesis.
The hypothesis is motivated by C9 feature use, not proven as the actual TX height.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,time,numpy as np
R=Path(__file__).resolve().parents[1];D=R/'features/c9_points';O=R/'features/c10_points';O.mkdir(exist_ok=True)
meta=json.loads((D/'metadata.json').read_text())
names=[f'phase0_margin_bin_{k}' for k in range(8)]+['phase0_buried_length','phase0_positive_fraction','phase0_max_location','phase0_critical_RX','phase0_crossings','phase0_TX_height']
meta['features']=meta['features']+names
t=(np.arange(128,dtype=np.float32)+.5)/128;tt=np.r_[np.float32(0),t,np.float32(1)];dt=np.diff(tt)
start=time.time()
for part in ['fit','dev']:
 x=np.load(D/f'{part}_x.npy',mmap_mode='r');assert not (O/f'{part}_x.npy').exists()
 out=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float32,shape=(len(x),len(meta['features'])))
 for lo in range(0,len(x),16384):
  sl=slice(lo,min(lo+16384,len(x)));v=x[sl];frac=np.remainder(v[:,31],1);e=2*np.sinh(v[:,104:232]*3)+frac[:,None]*(1-t)
  bins=e.reshape(-1,8,16).max(2);ep=np.concatenate((-(3-frac[:,None]),e,np.full((len(v),1),-1,np.float32)),1)
  a,z=ep[:,:-1],ep[:,1:];length=((np.maximum(a,0)+np.maximum(z,0))/np.maximum(abs(a)+abs(z),1e-9)*dt).sum(1)
  dist=np.sqrt((v[:,5]*128)**2+(v[:,6]*128)**2+(v[:,7]*50+frac)**2)
  extras=np.column_stack((np.arcsinh(bins/2)/3,np.log1p(length*dist)/6,(e>0).mean(1),t[e.argmax(1)],np.arcsinh((e/t).max(1)/2)/3,((a>0)!=(z>0)).sum(1),3-frac))
  out[sl]=np.concatenate((v,extras.astype(np.float32)),1)
 out.flush();print(part,time.time()-start,flush=True)
meta['hypothesis']='Phase-zero raster TX rounding; no assertion of recovered actual TX. All new path features derived only from input H/TX.';(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

