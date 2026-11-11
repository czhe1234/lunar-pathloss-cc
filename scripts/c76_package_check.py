"""Replay the exported CLI on raw inputs and compare it with saved official prediction shards."""
import csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c72_continuation'
def main():
    release=R/'releases/C76_20261103';run=R/'runs/c76_selected'
    assert (release/'infer.py').is_file()
    rows=list(csv.DictReader((R.parent/'data/val_index.csv').open()));hh=np.load(R.parent/'data/val_hm.npy',mmap_mode='r')
    out=A/'release_replay';out.mkdir(exist_ok=True);report=[]
    jobs=[]
    for card,shard,j in [(1,0,21),(4,2,33)]:
        d=run/'predictions'/f'val_full_c34_fresh_shard{shard}of3';i=json.load(open(d/'indices.json'))['index_rows'][j];row=rows[i]
        h=hh[int(row['hm_row'])];inp=out/f'input_{i}.npz';pred=out/f'prediction_{i}.npy'
        np.savez(inp,heightmap=h,tx_row=int(row['tx_row']),tx_col=int(row['tx_col']))
        log=open(R/f'logs/c76_package_card{card}.log','w')
        cmd=['./run_gpu.sh',str(card),str(release/'infer.py'),'--input',str(inp),'--output',str(pred)]
        jobs.append((subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT),log,d,i,j,pred,card))
    for process,log,d,i,j,pred,card in jobs:
        code=process.wait();log.close();assert code==0,(card,code)
        v=np.load(pred);baseline=np.stack([np.load(d/f'pl{b}.npy',mmap_mode='r')[j] for b in ['415','58']])
        delta=v.astype(np.float64)-baseline;rm=np.sqrt((delta*delta).mean((1,2)))
        assert v.shape==(2,256,256) and np.isfinite(v).all() and rm.max()<.015,(card,rm)
        report.append(dict(card=card,validation_index=i,RMSE_vs_exported_maps_dB=rm.tolist(),max_abs_dB=np.abs(delta).max((1,2)).tolist()))
    (A/'package_replay.json').write_text(json.dumps(dict(verified=True,scope='Exported release CLI with raw H and integer TX only; compare to predictions, no validation targets in replay.',records=report),indent=2)+'\n')
    with open(R/'logs/c76_slow_map_replay.log','w') as log:
        subprocess.run(['./run_gpu.sh','1','src/c76_retime.py'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
