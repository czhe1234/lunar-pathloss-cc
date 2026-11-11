"""Measure the normal raw-input interface on full TRAIN-side maps and cross-check against independent components."""
import argparse,csv,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c74_inference import Predictor
from src.c67_dense import frozen_maps
import numpy as np
import torch
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--reference-graph');a=p.parse_args()
    gpu=gpu_guard();out=R/a.out;out.mkdir(parents=True,exist_ok=True)
    conf=R/a.model;net=Predictor(conf);w=net.weight
    post_run=R/'runs'/net.cfg.get('internal_local_run','c72_observed')/'full_internal';ids=json.load(open(post_run/'indices.json'))
    chosen=np.random.default_rng(74086).choice(len(ids),8,replace=False).tolist()
    refpost=np.load(post_run/'correction.npy',mmap_mode='r');rows=list(csv.DictReader((DATA/'train_index.csv').open()));height=np.load(DATA/'train_hm.npy',mmap_mode='r')
    graphs=np.load(R/a.reference_graph/'prediction.npy',mmap_mode='r') if net.need_graph else None
    if net.need_graph:assert json.load(open(R/a.reference_graph/'indices.json'))==ids
    reference=frozen_maps([ids[i] for i in chosen],'dev')
    features={}
    if net.gate is not None:
        for f in ('c73_compact','c73_dev_extra_compact'):
            for folder in (R/'features'/f).glob('part*'):
                v=np.load(folder/'features.npy',mmap_mode='r')
                features.update({i:(v,j) for j,i in enumerate(json.load(open(folder/'indices.json')))})
    records=[];times=[]
    for k,pos in enumerate(chosen):
        i=ids[pos];row=rows[i];h=height[int(row['hm_row'])];tr,tc=int(row['tx_row']),int(row['tx_col'])
        alpha=w
        if net.gate is not None:
            src,j=features[i];vis=src[j,16];gg=np.asarray(net.cfg['gate_weights'])
            alpha=np.stack([gg[b][np.where(vis>.99,0,np.where(vis>=.01,1,np.where(src[j,41+b]<-.8,3,2)))] for b in range(2)])
        ref=(1-alpha)*(reference[k]+refpost[pos].astype(np.float32))
        if net.need_graph:ref+=alpha*graphs[pos].astype(np.float32)
        local=[]
        for rep in range(3 if k==0 else 2):
            torch.cuda.synchronize();start=time.perf_counter();v=net.predict(h,tr,tc);torch.cuda.synchronize();dt=time.perf_counter()-start
            local.append(dt)
            if k or rep:times.append(dt)
        diff=np.sqrt(np.mean((v.astype(np.float64)-ref)**2,axis=(1,2)))
        assert np.isfinite(v).all() and max(diff)<.25,(i,diff)
        records.append(dict(index=i,seconds=local,cached_replay_RMS_diff=diff.tolist()))
        print(records[-1],flush=True)
    report=dict(hardware=gpu,warm_mean_seconds=float(np.mean(times)),warm_p95_seconds=float(np.quantile(times,.95)),cold_first_seconds=records[0]['seconds'][0],
        warm_max_seconds=max(times),maps=len(chosen),warm_calls=len(times),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,records=records,under_500ms_mean=np.mean(times).item()<.5)
    atomic_json(out/'result.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
