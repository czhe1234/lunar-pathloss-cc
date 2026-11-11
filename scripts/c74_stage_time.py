"""A800 stage latency on actual TRAIN inputs. Exploratory timing, not quality selection."""
import sys,time,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c74_inference import Predictor
import torch
import numpy as np
gpu_guard();net=Predictor(R/'audit/c72_continuation/pilot_interface')
rows=list(csv.DictReader((DATA/'train_index.csv').open()));hh=np.load(DATA/'train_hm.npy',mmap_mode='r')
times=[]
for i in [234,0,1700,2997,12630]:
    row=rows[i];h=hh[int(row['hm_row'])];r,c=int(row['tx_row']),int(row['tx_col']);out={}
    handlers=[]
    for obj,key,alias in [(net.base,'features','landform'),(net.base.teacher,'forward','teacher'),(net.base.low,'forward','profile'),(net.base.high,'forward','surface'),(net.features.geo,'native','nativegeom'),(net.features.paths,'raw','raw'),(net.post,'forward','post')]:
        orig=getattr(obj,key)
        def timed(*a,_orig=orig,_alias=alias,**kw):
            torch.cuda.synchronize();t=time.perf_counter();v=_orig(*a,**kw);torch.cuda.synchronize();out[_alias]=out.get(_alias,0)+time.perf_counter()-t;return v
        setattr(obj,key,timed);handlers.append((obj,key,orig))
    start=time.perf_counter();net.predict(h,r,c);out['total']=time.perf_counter()-start
    for obj,key,orig in handlers:setattr(obj,key,orig)
    times.append(out);print(i,out,flush=True)
atomic_json(R/'audit/c72_continuation/pilot_A800/stages.json',times)
