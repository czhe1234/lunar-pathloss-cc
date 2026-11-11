"""Prepare uniform 64-pixel/map full official TRAIN data for fixed C9 refit.
Never reads official VAL. No new labels or pseudo-labels.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,time,numpy as np
R=Path(__file__).resolve().parents[1];D=R/'features/c10_points';L=R/'features/c6_points';O=R/'features/c10_full_points';O.mkdir(exist_ok=True)
meta=json.loads((D/'metadata.json').read_text());fitids=np.load(L/'fit_map_indices.npy');devids=np.load(L/'dev_map_indices.npy');assert len(set(fitids)&set(devids))==0
ids=np.r_[fitids,devids];assert len(ids)==25720 and np.array_equal(np.sort(ids),np.arange(25720))
n=len(ids)*64;shapes={'x':(n,len(meta['features'])),'y':(n,2),'mask':(n,2),'fspl':(n,2),'pixel':(n,)}
start=time.time();assert not (O/'complete.json').exists()
for name,shape in shapes.items():
 srcD=D if name=='x' else L;arrfit=np.load(srcD/f'fit_{name}.npy',mmap_mode='r');arrdev=np.load(srcD/f'dev_{name}.npy',mmap_mode='r')
 out=np.lib.format.open_memmap(O/f'{name}.npy',mode='w+',dtype=arrfit.dtype,shape=shape)
 out[:len(arrfit)]=arrfit
 for j in range(len(devids)):out[len(arrfit)+j*64:len(arrfit)+(j+1)*64]=arrdev[j*256:j*256+64]
 out.flush();print(name,time.time()-start,flush=True)
np.save(O/'map_indices.npy',ids);meta.update({'source_split':'official train, all 25720 maps / 6120 terrains','points_per_map':64,'points':n,'DEV_reuse':'After method selection, include first64 uniformly sampled DEV points per map in full TRAIN refit. Official validation untouched.','no_official_val_loaded':True})
(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');(O/'complete.json').write_text(json.dumps({'complete':True,'seconds':time.time()-start})+'\n')

