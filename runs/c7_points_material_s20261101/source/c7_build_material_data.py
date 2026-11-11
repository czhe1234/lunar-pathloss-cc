"""Material descriptor contrast; same point selection as C6. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,csv,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c7_material_features import features,NAMES
D=R/'features/c6_points';O=R/'features/c7_points';O.mkdir(exist_ok=True)
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
meta=json.loads((D/'metadata.json').read_text());meta.update({'added_features':NAMES,'c7_source_hash':hashlib.sha256((R/'src/c7_material_features.py').read_bytes()).hexdigest(),'features':meta['features'][:31]+NAMES,'source_inputs':'Heightmap and TX only. No labels, masks or baseline predictions read by this feature script.','material':'Published fixed 10m slab parameters. Approximate bilinear-surface intersections, not original mesh and not a propagation simulation.'})
start=time.time()
for part in ['fit','dev']:
 ids=np.load(D/f'{part}_map_indices.npy');pixels=np.load(D/f'{part}_pixel.npy',mmap_mode='r').reshape(len(ids),-1);n=pixels.shape[1];x=np.load(D/f'{part}_x.npy',mmap_mode='r')
 assert not (O/f'{part}_complete.json').exists()
 out=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float32,shape=(len(ids)*n,len(meta['features'])))
 def item(j):
  i=int(ids[j]);r=rows[i]
  return j,features(H[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']),pixels[j])
 with ThreadPoolExecutor(max_workers=8) as pool:
  for lo in range(0,len(ids),128):
   for j,f in pool.map(item,range(lo,min(lo+128,len(ids)))):
    sl=slice(j*n,(j+1)*n);out[sl,:31]=x[sl,:31];out[sl,31:]=f
   if lo%2048==0:print(json.dumps({'part':part,'maps':min(lo+128,len(ids)),'seconds':time.time()-start}),flush=True)
 out.flush();(O/f'{part}_complete.json').write_text(json.dumps({'maps':len(ids),'points':len(out),'complete':True})+'\n')
meta['seconds']=time.time()-start;(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'complete':True,'seconds':meta['seconds']}),flush=True)

