import time,json,os,concurrent.futures
from pathlib import Path
import numpy as np
from c21_data import Data,phase,R
out=R/'features/c21_phase';out.mkdir(exist_ok=True);d=Data(False)
dest=np.lib.format.open_memmap(out/'train.npy',mode='w+',dtype=np.float16,shape=(len(d.rows),14,256,256))
done=np.lib.format.open_memmap(out/'ready.npy',mode='w+',dtype=np.uint8,shape=(len(d.rows),));done[:]=0
ids=d.ids['fit'].copy();np.random.default_rng(20261123).shuffle(ids);ids=np.concatenate((ids,d.ids['dev']))
t0=time.time()
def work(i):
 row=d.rows[int(i)];h=d.H[int(row['hm_row'])];t=time.time();p=phase(h,int(row['tx_row']),int(row['tx_col']))
 dest[i]=p;done[i]=1;return time.time()-t
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for n,sec in enumerate(pool.map(work,ids,chunksize=1),1):
  if n%100==0:
   row={'completed':n,'total':len(ids),'seconds':time.time()-t0,'last_map_seconds':sec};print(json.dumps(row),flush=True)
   (out/'progress.json').write_text(json.dumps(row))
dest.flush();done.flush();(out/'complete.json').write_text(json.dumps({'maps':len(ids),'seconds':time.time()-t0,'source':'published H andTX, phase14 without inverse hyperbolic roundtrip','inputs_only':True}))
