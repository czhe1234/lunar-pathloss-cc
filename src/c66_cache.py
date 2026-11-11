"""Input-only, terrain-held-out protocol. Cache the frozen C54 baseline on FIT maps."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard, atomic_json
from src.c60_inference import Predictor
import numpy as np
import torch

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--of',type=int,default=2)
    args=parser.parse_args()
    gpu=gpu_guard()
    selection=json.loads((ROOT/'audit/c66_transport/panels.json').read_text())
    indices=selection['fit_maps'][args.shard::args.of]
    rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()))
    heights=np.load(ROOT.parent/'data/train_hm.npy',mmap_mode='r')
    model_dir=ROOT/'runs/c54_combo_dev'
    predictor=Predictor(model_dir)
    dest=ROOT/f'features/c66_frozen_maps/fit{args.shard}of{args.of}'
    dest.mkdir(parents=True,exist_ok=False)
    preds=np.lib.format.open_memmap(dest/'prediction.npy',mode='w+',dtype=np.float32,shape=(len(indices),2,256,256))
    atomic_json(dest/'indices.json',indices)
    start=time.monotonic()
    for j,idx in enumerate(indices):
        row=rows[idx]
        preds[j]=predictor.predict(heights[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']))
        if j%50==0 or j+1==len(indices):
            progress={'maps':j+1,'total':len(indices),'seconds':time.monotonic()-start}
            atomic_json(dest/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    preds.flush()
    atomic_json(dest/'complete.json',{'complete':True,'maps':len(indices),'gpu':gpu,'input_only':True,'model_manifest_sha256':hashlib.sha256((model_dir/'manifest.json').read_bytes()).hexdigest()})
