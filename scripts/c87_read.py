"""Read frozen released-validation predictions in original file order with original masks."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'data'
OUT=ROOT/'audit/c87_val'
OUT.mkdir(exist_ok=True)
def maps(want=None):
    rows=list(csv.DictReader((DATA/'val_index.csv').open()))
    y=[np.load(DATA/f'val_pl{b}.npy',mmap_mode='r') for b in ('415','58')]
    mask=[np.load(DATA/f'val_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
    seen=set()
    for d in sorted((ROOT/'runs/c82_selected/predictions').glob('val_full*')):
        ids=json.load(open(d/'indices.json'));meta=json.load(open(d/'inference.json'))
        assert meta['complete'] and not meta['target_or_masks_used']
        preds=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ('415','58')]
        for j,(i,sid) in enumerate(zip(ids['index_rows'],ids['sample_ids'])):
            assert i not in seen and rows[i]['sample_id']==sid
            seen.add(i)
            if want is not None and not want(i):continue
            yield i,rows[i],np.stack([ar[i] for ar in y]).astype(np.float32),np.stack([ar[j] for ar in preds]),np.stack([np.unpackbits(ar[i]).reshape(256,256) for ar in mask]).astype(bool)
    assert seen==set(range(len(rows))) and len(rows)==2330
def write(name,x):
    dest=OUT/name;tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda t:t.item())+'\n');tmp.replace(dest)
