"""Replay raw maps from the installed release on authorized hardware; no validation target read."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];PKG=R/'releases/C82_20261103'
sys.path.insert(0,str(PKG))
from src.c82_inference import Predictor
import torch
from src.cc_core import gpu_guard
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--card',type=int,default=3);a=pa.parse_args()
    checked=gpu_guard();net=Predictor(PKG)
    data=R.parent/'data';rows=list(csv.DictReader((data/'val_index.csv').open()));hs=np.load(data/'val_hm.npy',mmap_mode='r')
    pp={}
    for dest in (R/'runs/c82_selected/predictions').glob('val*'):
        ix=json.load(open(dest/'indices.json'))['index_rows']
        ar=[np.load(dest/f'pl{b}.npy',mmap_mode='r') for b in ('415','58')]
        for j,i in enumerate(ix):pp[i]=(ar,j)
    # Distinct terrains and coordinates, including one index from each official shard.
    indices=[19,401,781,977,1104,1288,1519,1703,1854,2003,2207,2318]
    events=[]
    for warm_pass in range(2):
        for i in indices:
            row=rows[i];h=hs[int(row['hm_row'])];tr,tc=int(row['tx_row']),int(row['tx_col'])
            torch.cuda.synchronize();start=time.monotonic();out=net.predict(h,tr,tc);torch.cuda.synchronize();dt=time.monotonic()-start
            ar,j=pp[i];ref=np.stack([x[j] for x in ar])
            err=out.astype('float64')-ref;rm=np.sqrt((err*err).mean((1,2)))
            assert out.shape==(2,256,256) and np.isfinite(out).all() and max(rm)<.02,(i,rm)
            events.append(dict(map=i,pass_index=warm_pass,seconds=dt,rmse_vs_stored_by_band=rm.tolist(),max_abs=np.abs(err).max(axis=(1,2)).tolist()))
    v=np.array([z['seconds'] for z in events[1:]])
    rpt=dict(gpu=checked,maps=len(indices),calls=len(events),warm_calls=len(v),mean_s=v.mean(),p95_s=np.quantile(v,.95),
             max_s=v.max(),warm_fraction_over_500ms=float((v>.5).mean()),cold_s=events[0]['seconds'],rows=events,
             input='raw released H and integer TX; targets and actual masks never opened by this script',
             location='isolated package imports from releases/C82_20261103')
    path=R/f'audit/c79_receiver/package_replay_gpu{a.card}.json';path.write_text(json.dumps(rpt,indent=2)+'\n')
    print('warm mean/p95/max',v.mean(),np.quantile(v,.95),v.max(),'rmse drift max',max(max(z['rmse_vs_stored_by_band']) for z in events),flush=True)
if __name__=='__main__':main()
