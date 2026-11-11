"""Read previously frozen predictions and masks only to form the diagnostic reference panel."""
from pathlib import Path
import json,sys,time
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear';split=sys.argv[1];panel=np.load(A/f'{split}_panel.npz');ids=panel['ids'];pp=panel['pixels']
def load(p):return np.load(p,mmap_mode='r',allow_pickle=False)
m=[load(R.parent/f'data/{split}_mask{b}.npy') for b in ('415','58')];y=[load(R.parent/f'data/{split}_pl{b}.npy') for b in ('415','58')]
if split=='train':
    base=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
    assert list(ids)==json.load(open(base/'indices.json'))['index_rows']
    direct=[load(base/f'pl{b}.npy') for b in ('415','58')];post=load(R/'runs/c72_observed/full_internal/correction.npy')
    graph=load(R/'runs/c73_wide_real/full_internal/prediction.npy')
    lookup={}
    for name in ('c73_compact','c73_dev_extra_compact'):
        for d in (R/'features'/name).glob('part*'):
            feature=load(d/'features.npy');lookup.update({i:(feature,j) for j,i in enumerate(json.load(open(d/'indices.json')))})
else:
    lookup={}
    for d in sorted((R/'runs/c76_selected/predictions').glob('*')):
        ar=[load(d/f'pl{b}.npy') for b in ('415','58')]
        lookup.update({i:(ar,j) for j,i in enumerate(json.load(open(d/'indices.json'))['index_rows'])})
out=np.lib.format.open_memmap(A/f'{split}_reference.npy','w+',dtype='float32',shape=(len(ids),pp.shape[1],7))
for j,i in enumerate(ids):
    loc=pp[j];out[j,:,0:2]=np.stack([v[i].ravel()[loc] for v in y],1)
    out[j,:,4:6]=np.stack([np.unpackbits(v[i])[loc] for v in m],1)
    ar,k=lookup[i]
    if split=='train':
        a=np.stack([v[j].ravel()[loc] for v in direct],1)
        output=.5*(a+post[j].reshape(2,-1)[:,loc].T.astype('float32')+graph[j].reshape(2,-1)[:,loc].T)
        out[j,:,6]=ar[k,16].ravel()[loc]
    else:
        output=np.stack([v[k].ravel()[loc] for v in ar],1);out[j,:,6]=-1
    out[j,:,2:4]=output
out.flush()
(A/f'{split}_reference.json').write_text(json.dumps(dict(shape=list(out.shape),columns=['Y415','Y58','C76_415','C76_58','valid415','valid58','cached_C76_visible_fraction_if_internal'],scope='C76 research counterpart' if split=='train' else 'actual deployed frozen C76'),indent=2)+'\n')
print('saved',split,out.shape)
