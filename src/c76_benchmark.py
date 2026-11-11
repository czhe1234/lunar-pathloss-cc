"""Benchmark the selected full-TRAIN version on distinct TRAIN-side inputs and verify its portable release if requested."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,atomic_json,gpu_guard
from src.c74_inference import Predictor
import numpy as np
import torch
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='runs/c76_selected');p.add_argument('--out',required=True);a=p.parse_args()
    d=R/a.model;gpu=gpu_guard();net=Predictor(d);rows=list(csv.DictReader((DATA/'train_index.csv').open()));hh=np.load(DATA/'train_hm.npy',mmap_mode='r')
    selected=np.random.default_rng(76083).choice(len(rows),12,replace=False).tolist();times=[];records=[];cold=0.
    for j,i in enumerate(selected):
        row=rows[i];h=hh[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col']);local=[]
        for rep in range(3 if j==0 else 2):
            torch.cuda.synchronize();t=time.perf_counter();m=net.predict(h,tr,tc);torch.cuda.synchronize();dt=time.perf_counter()-t
            if j or rep:times.append(dt)
            else:cold=dt
            local.append(dt)
        assert np.isfinite(m).all() and m.shape==(2,256,256)
        records.append(dict(i=i,latency_s=local,digest=hashlib.sha256(m.tobytes()).hexdigest()))
    dest=R/a.out;dest.mkdir(parents=True,exist_ok=True)
    info=dict(gpu=gpu,warm_mean=float(np.mean(times)),warm_p50=float(np.median(times)),warm_p95=float(np.quantile(times,.95)),warm_max=max(times),warm_calls=len(times),cold=cold,maps=len(selected),records=records,all_warm_under_500ms=max(times)<=.5,
        scope='Actual frozen full-TRAIN base with selected corrections, raw TRAIN height and TX. Timed both output maps together; no official hardware run.')
    atomic_json(dest/'result.json',info);print(json.dumps(info),flush=True)
if __name__=='__main__':main()
