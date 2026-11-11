"""Generate full fixed-VAL maps from a frozen standalone raw-height/TX interface. No targets or masks."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c74_inference import Predictor
import numpy as np
import torch
def main(predictor_cls=Predictor):
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--shard',required=True,type=int);p.add_argument('--of',type=int,default=3)
    a=p.parse_args();gpu=gpu_guard();run=R/'runs'/a.model;frozen=json.load(open(run/'freeze.json'))
    assert not frozen['official_VAL_used_for_selection']
    sha=lambda p:hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
    for name,digest in frozen['files'].items():assert sha(run/name)==digest,name
    for name,digest in frozen['implementation'].items():assert sha(R/'src'/name)==digest,name
    rows=list(csv.DictReader((DATA/'val_index.csv').open()));hmap=np.load(DATA/'val_hm.npy',mmap_mode='r')
    assert len(rows)==2330 and len({r['terrain_id'] for r in rows})==680
    cuts=[0,900,1800,2330];ix=list(range(cuts[a.shard],cuts[a.shard+1])) if a.of==3 else list(range(len(rows)))[a.shard::a.of]
    out=run/'predictions'/f'val_full_c34_fresh_shard{a.shard}of{a.of}';out.mkdir(parents=True,exist_ok=True)
    check=sha(run/'freeze.json');manifest=dict(index_rows=ix,sample_ids=[rows[i]['sample_id'] for i in ix],source_split='val',freeze_sha256=check)
    if (out/'indices.json').exists():assert json.load(open(out/'indices.json'))==manifest
    else:atomic_json(out/'indices.json',manifest)
    if (out/'inference.json').exists():
        assert json.load(open(out/'inference.json'))['complete'];return
    pos=0
    if (out/'progress.json').exists():pos=int(json.load(open(out/'progress.json'))['maps'])
    ar=[]
    for b in ['415','58']:
        dest=out/f'pl{b}.npy';mode='r+' if dest.exists() else 'w+'
        if not dest.exists():assert pos==0
        arr=np.lib.format.open_memmap(dest,mode=mode,dtype=np.float32,shape=(len(ix),256,256));assert arr.shape==(len(ix),256,256)
        ar.append(arr)
    timing=out/'seconds_per_map.npy';tm='r+' if timing.exists() else 'w+'
    elapsed=np.lib.format.open_memmap(timing,mode=tm,dtype=np.float64,shape=(len(ix),))
    if tm=='w+':elapsed[:]=np.nan
    net=predictor_cls(run);times=[];start=time.monotonic()
    for j in range(pos,len(ix)):
        row=rows[ix[j]];height=hmap[int(row['hm_row'])]
        torch.cuda.synchronize();t=time.perf_counter();v=net.predict(height,int(row['tx_row']),int(row['tx_col']));torch.cuda.synchronize();times.append(time.perf_counter()-t)
        elapsed[j]=times[-1]
        assert np.isfinite(v).all() and v.shape==(2,256,256)
        for band in range(2):ar[band][j]=v[band]
        if (j+1)%40==0 or j+1==len(ix):
            for arr in ar:arr.flush()
            elapsed.flush()
            rr=dict(maps=j+1,total=len(ix),seconds=time.monotonic()-start,last_seconds=times[-1],freeze_sha256=check)
            atomic_json(out/'progress.json',rr);print(json.dumps(rr),flush=True)
    assert np.isfinite(elapsed).all()
    warm=times[1:] or times
    atomic_json(out/'inference.json',dict(complete=True,maps=len(ix),source='raw released VAL height and TX only',target_or_masks_used=False,
        gpu=gpu,freeze_sha256=check,resumed_at=pos,warm_mean=float(np.mean(warm)) if warm else None,warm_p95=float(np.quantile(warm,.95)) if warm else None,
        first_seconds=times[0] if times else None,elapsed_seconds=time.monotonic()-start,all_map_count=len(ix),all_over_500ms=int((elapsed>.5).sum()),warm_run_over_500ms=int((np.asarray(warm)>.5).sum()),warm_run_count=len(warm),warm_max=max(warm) if warm else None))
if __name__=='__main__':main()
