"""Consistent fractional-height hypothesis for direct and surrounding geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,csv,json,time,hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c12_surface_features import features as surface,terrain_fields,tx_visibility,BASE as SB
from src.c12_material_features import features as material,BASE as MB
from src.c7_surface_features import features as original_surface
from src.c7_material_features import features as original_material
D=R/'features/c6_points';C=R/'features/c10_points';O=R/'features/c12_points';O.mkdir(exist_ok=True)
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r');rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
meta=json.loads((C/'metadata.json').read_text());oldn=len(meta['features'])
meta['features'] += ['phase0_surface_'+n for n in SB]+['phase0_material_'+n for n in MB+['slab_TE_415','slab_TM_415','slab_TE_58','slab_TM_58','slab_mix_415','slab_mix_58']]
meta.update(C12_hypothesis='Use 3-frac(H_TX) consistently for surface visibility, incidence and crossing geometry. Not a recovered actual coordinate.',no_target_or_mask_read_by_builder=True)
@lru_cache(maxsize=64)
def fields(i):return terrain_fields(H[i])
checks=[]
ids0=np.load(D/'fit_map_indices.npy');pix0=np.load(D/'fit_pixel.npy',mmap_mode='r').reshape(len(ids0),-1)
for j in [0,10,100,1000]:
 i=int(ids0[j]);row=rows[i];hi=int(row['hm_row']);tr=int(row['tx_row']);tc=int(row['tx_col']);pix=pix0[j];vis=[tx_visibility(G[i],ht) for ht in [3,2]]
 a=surface(H[hi],fields(hi),[vis[0]],tr,tc,pix,txheight=3.);b=original_surface(H[hi],fields(hi),vis,tr,tc,pix)[:,:20]
 c=material(H[hi],tr,tc,pix,txheight=3.);d=original_material(H[hi],tr,tc,pix)[:,:16]
 assert np.array_equal(a,b) and np.array_equal(c,d)
 checks.append({'index_row':i,'surface_maxdiff':float(abs(a-b).max()),'material_maxdiff':float(abs(c-d).max())})
(R/'audit/c12_feature_replay_check.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
start=time.time()
for part in ['fit','dev']:
 ids=np.load(D/f'{part}_map_indices.npy');pixels=np.load(D/f'{part}_pixel.npy',mmap_mode='r').reshape(len(ids),-1);n=pixels.shape[1];x=np.load(C/f'{part}_x.npy',mmap_mode='r')
 assert not (O/f'{part}_x.npy').exists()
 out=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float32,shape=(len(x),len(meta['features'])))
 def item(j):
  i=int(ids[j]);row=rows[i];hi=int(row['hm_row']);tr=int(row['tx_row']);tc=int(row['tx_col']);h=H[hi];txh=3.-float(np.remainder(h[tr,tc],1.))
  vv=[tx_visibility(G[i],txh)]
  f=np.concatenate((surface(h,fields(hi),vv,tr,tc,pixels[j],txheight=txh),material(h,tr,tc,pixels[j],txheight=txh)),1)
  assert np.isfinite(f).all()
  return j,f
 with ThreadPoolExecutor(max_workers=8) as pool:
  for lo in range(0,len(ids),128):
   for j,f in pool.map(item,range(lo,min(lo+128,len(ids)))):
    sl=slice(j*n,(j+1)*n);out[sl,:oldn]=x[sl];out[sl,oldn:]=f
   if lo%2048==0:print(json.dumps({'part':part,'maps_done':min(lo+128,len(ids)),'total':len(ids),'seconds':time.time()-start}),flush=True)
 out.flush();(O/f'{part}_complete.json').write_text(json.dumps({'complete':True,'maps':len(ids),'points':len(x)})+'\n')
meta['seconds']=time.time()-start;(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'complete':True,'seconds':meta['seconds']}),flush=True)
