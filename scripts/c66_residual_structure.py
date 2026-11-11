"""Localize C65-family error on internal terrain holdout using the frozen C54 counterpart."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
rows = list(csv.DictReader((DATA / "train_index.csv").open()))
directory = ROOT / "runs/c54_combo_dev/bf16_compiled/all_shard0of1"
indices = json.loads((directory / "indices.json").read_text())["index_rows"]
predictions = [np.load(directory / f"pl{band}.npy", mmap_mode="r") for band in ["415", "58"]]
target = [np.load(DATA / f"train_pl{band}.npy", mmap_mode="r") for band in ["415", "58"]]
masks = [np.load(DATA / f"train_mask{band}.npy", mmap_mode="r") for band in ["415", "58"]]
heights = np.load(DATA / "train_hm.npy", mmap_mode="r")
geo = np.load(ROOT / "features/c4_extra/train.npy", mmap_mode="r")
yy, xx = np.mgrid[:256,:256]
acc = {}
def record(key, band, error, subset):
    if key not in acc:
        acc[key] = np.zeros((2,4), dtype=float)
    e = error[subset].astype('float64')
    acc[key][band] += [len(e), np.sum(e*e), np.sum(e), np.sum(e[e>0]**2)]
for pos, idx in enumerate(indices):
    row = rows[idx]
    height = heights[int(row['hm_row'])]
    tr,tc = int(row['tx_row']),int(row['tx_col'])
    tx_h = height[tr,tc]
    critical = 2.5 + 2*np.sinh(3*geo[idx,0].astype('float32'))
    gap = critical - (3 - np.remainder(tx_h,1))
    zones = {'clear':gap<-.4,'transition':(gap>=-.4)&(gap<=.4),'shadow':gap>.4}
    for b in range(2):
        y = np.asarray(target[b][idx],dtype='float32')
        pred = predictions[b][pos]
        mask = np.unpackbits(masks[b][idx]).reshape(256,256).astype(bool)
        error = y - pred
        record('all',b,error,mask)
        for key,z in zones.items():
            record(key,b,error,mask&z)
        # Exact nonoverlapping 4x4 decomposition into tile mean and within-tile variations.
        z = np.where(mask,error,0).reshape(64,4,64,4)
        n = mask.reshape(64,4,64,4).sum((1,3))
        local = z.sum((1,3))/np.maximum(n,1)
        mean = np.repeat(np.repeat(local,4,0),4,1)
        record('block4_mean',b,mean,mask)
        record('within_block4',b,error-mean,mask)
all_sse = acc['all'][:,1]
metrics = {name:{'count':v[:,0].astype(int).tolist(), 'rmse':np.sqrt(v[:,1]/v[:,0]).tolist(),'sse_fraction':(v[:,1]/all_sse).tolist(),'larger_true_PL_sse_fraction':(v[:,3]/v[:,1]).tolist()} for name,v in acc.items()}
assert np.max(np.abs(np.array(metrics['block4_mean']['sse_fraction'])+metrics['within_block4']['sse_fraction']-1))<1e-6
output = {'scope':'All 612 internal DEV terrains, 2562 maps, C54 internal counterpart. Masks only for scoring. No official VAL.', 'error':'label minus prediction, in dB', 'metrics':metrics}
dest=ROOT/'audit/c66_transport';dest.mkdir(exist_ok=True)
(dest/'residual_structure.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output),flush=True)
