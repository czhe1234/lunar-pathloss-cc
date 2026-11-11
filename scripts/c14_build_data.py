"""Same fixed pixels, receiver-only new input descriptors; never read labels or masks."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,csv,time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c14_receiver_geometry import fields,features,NAMES
D=R/'features/c6_points';O=R/'features/c14_receiver_points';O.mkdir(exist_ok=False)
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
@lru_cache(maxsize=64)
def terrain(hi):return fields(H[hi])
start=time.time()
for part in ['fit','dev']:
 ids=np.load(D/f'{part}_map_indices.npy');pixels=np.load(D/f'{part}_pixel.npy',mmap_mode='r').reshape(len(ids),-1);n=pixels.shape[1]
 out=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float32,shape=(len(ids)*n,len(NAMES)))
 def item(j):
  r=rows[int(ids[j])]
  return j,features(terrain(int(r['hm_row'])),int(r['tx_row']),int(r['tx_col']),pixels[j])
 with ThreadPoolExecutor(max_workers=8) as pool:
  for lo in range(0,len(ids),128):
   for j,f in pool.map(item,range(lo,min(lo+128,len(ids)))):out[j*n:(j+1)*n]=f
   if lo%4096==0:print(json.dumps({'part':part,'done':min(lo+128,len(ids)),'total':len(ids),'seconds':time.time()-start}),flush=True)
 out.flush()
meta={'features':NAMES,'source_inputs':'Original H and TX only, same C6 FIT/DEV pixel indices','no_targets_or_masks_read':True,'cell_geometry':'five raw-height RX points at center and +/-0.5m corners, TX uses phase-zero floor(Htx)+3 hypothesis; no actual cell recovery claim','seconds':time.time()-start}
(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
