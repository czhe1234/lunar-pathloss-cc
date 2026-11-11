import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from functools import lru_cache
import json,csv,time,sys
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c8_tx_context import terrain_fields,tx_features,NAMES
D=R/'features/c6_points';O=R/'features/c8_points';O.mkdir(exist_ok=True)
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
meta=json.loads((D/'metadata.json').read_text());meta['features']=meta['features'][:31]+NAMES;meta['C8_context']='Transmitter neighborhood and height phase. Not an assertion about actual TX height; inputs only.'
@lru_cache(maxsize=64)
def fields(i):return terrain_fields(H[i])
start=time.time()
for part in ['fit','dev']:
 ids=np.load(D/f'{part}_map_indices.npy');x=np.load(D/f'{part}_x.npy',mmap_mode='r');n=len(x)//len(ids)
 assert not (O/f'{part}_complete.json').exists()
 out=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float32,shape=(len(x),len(meta['features'])))
 for j,i in enumerate(ids):
  r=rows[i];f=tx_features(fields(int(r['hm_row'])),int(r['tx_row']),int(r['tx_col']));sl=slice(j*n,(j+1)*n)
  out[sl,:31]=x[sl,:31];out[sl,31:]=f
 out.flush();(O/f'{part}_complete.json').write_text(json.dumps({'complete':True,'maps':len(ids),'seconds':time.time()-start})+'\n');print(part,time.time()-start,flush=True)
meta['seconds']=time.time()-start;(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

