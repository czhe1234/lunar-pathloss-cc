"""Repeat the single slow warm validation call on its original GPU. Keep the original timing."""
import csv,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,atomic_json,gpu_guard
from src.c74_inference import Predictor
import numpy as np
import torch
gpu=gpu_guard();run=R/'runs/c76_selected';net=Predictor(run)
rows=list(csv.DictReader((DATA/'val_index.csv').open()));hm=np.load(DATA/'val_hm.npy',mmap_mode='r')
records=[]
for folder in sorted((run/'predictions').glob('*')):
    meta=json.load(open(folder/'inference.json'))
    if meta['gpu']['uuid']!=gpu['uuid']:continue
    times=np.load(folder/'seconds_per_map.npy');ids=json.load(open(folder/'indices.json'))['index_rows']
    for pos in np.flatnonzero(times[1:]>.5)+1:
        i=ids[pos];neighbor=ids[min(len(ids)-1,pos+1)]
        def pred(j):
            row=rows[j];return net.predict(hm[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']))
        pred(neighbor);repeat=[]
        for _ in range(8):
            pred(neighbor);torch.cuda.synchronize();t=time.perf_counter();v=pred(i);torch.cuda.synchronize();repeat.append(time.perf_counter()-t)
        ref=np.stack([np.load(folder/f'pl{b}.npy',mmap_mode='r')[pos] for b in ['415','58']])
        rms=np.sqrt(((v.astype(np.float64)-ref)**2).mean((1,2)));assert max(rms)<.015
        records.append(dict(index=i,sample_id=rows[i]['sample_id'],original=float(times[pos]),repeat=repeat,mean=float(np.mean(repeat)),max=max(repeat),RMS_vs_original=rms.tolist()))
atomic_json(R/'audit/c72_continuation/latency_repeat.json',dict(gpu=gpu,records=records,scope='Post-hoc replay only. The original full-validation timing is retained.'));print(records,flush=True)
