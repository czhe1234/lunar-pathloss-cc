"""Compare the fused geometry computation with the original 294 features, then run full-map QA."""
import sys,time,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c74_features import FastDescriptors
from src.c47_inference import Predictor
import torch
import numpy as np
g=gpu_guard();old=Predictor.__new__(Predictor);yy,xx=np.mgrid[:256,:256];old.yy=yy.astype(np.float32);old.xx=xx.astype(np.float32);new=FastDescriptors()
rows=list(csv.DictReader((DATA/'train_index.csv').open()));hh=np.load(DATA/'train_hm.npy',mmap_mode='r');rec=[]
for i in [0,234,2701,9281,15752,24412]:
    r=rows[i];h=hh[int(r['hm_row'])];tr,tc=int(r['tx_row']),int(r['tx_col'])
    torch.cuda.synchronize();t=time.perf_counter();xo,fo=old.features(h,tr,tc);torch.cuda.synchronize();t1=time.perf_counter()
    xn,fn=new(h,tr,tc);torch.cuda.synchronize();t2=time.perf_counter()
    e=(xo-xn).abs();v=dict(idx=i,scalar_rmse=float((e*e).mean().sqrt()),non_surface_max=float(torch.cat((e[:,:232],e[:,272:]),1).max()),
        surface_mean=float(e[:,232:272].mean()),surface_max=float(e[:,232:272].max()),worst_channel=int(e.reshape(-1).argmax()%294),fs_max=float((fo-fn).abs().max()),legacy_seconds=t1-t,fused_seconds=t2-t1)
    rec.append(v);print(v,flush=True)
    assert v['fs_max']==0 and v['non_surface_max']<2e-5 and v['scalar_rmse']<.003,v
atomic_json(R/'audit/c72_continuation/descriptor_equivalence.json',dict(gpu=g,maps=rec))
