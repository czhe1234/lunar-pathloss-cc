"""Which scattering/visibility conditions still carry C69 error? Use only internal TRAIN-derived DEV."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];data=R.parent/'data'
ids=json.load(open(R/'runs/c69_registered_spatial/full_internal/indices.json'));folder=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
assert ids==json.load(open(folder/'indices.json'))['index_rows']
base=[np.load(folder/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
cor=np.load(R/'runs/c69_registered_spatial/full_internal/correction.npy',mmap_mode='r')
gt=[np.load(data/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];mw=[np.load(data/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
pp={}
for grp in ['c69_transport','c69_dev_extra_transport']:
    for d in (R/'features'/grp).glob('part*'):
        x=np.load(d/'features.npy',mmap_mode='r');pp.update({i:(x,j) for j,i in enumerate(json.load(open(d/'indices.json')))})
rng=np.random.default_rng(7331);acc={}
for j,i in enumerate(ids):
    ix=rng.choice(65536,2048,replace=False);x,pos=pp[i];visible=x[pos,10].reshape(-1)[ix].astype('float32')
    for b in range(2):
        m=np.unpackbits(mw[b][i])[ix].astype(bool);ref=gt[b][i].ravel()[ix].astype('float32')
        p=base[b][j].ravel()[ix];r=ref-(p+cor[j,b].ravel()[ix])
        one=x[pos,6+b].ravel()[ix].astype('float32')
        for name,cond in [('all',np.ones(len(ix),bool)),('clear',visible>.99),('partial', (visible>=.01)&(visible<=.99)),('shadow',visible<.01),
                          ('shadow_weak_first_order',(visible<.01)&(one<-.8)),('shadow_other',(visible<.01)&(one>=-.8))]:
            s=m&cond;e=r[s].astype('float64');key=(name,b)
            acc[key]=acc.get(key,np.zeros(3))+np.array([len(e),np.dot(e,e),e.sum()])
out={}
for (n,b),v in acc.items():
    out.setdefault(n,[]).append(dict(valid_count=int(v[0]),valid_share=v[0]/acc['all',b][0],SSE_share=v[1]/acc['all',b][1],RMSE=np.sqrt(v[1]/v[0]),bias=v[2]/v[0]))
d=dict(scope='All 612 internal DEV terrains, 2048 input-independent uniformly sampled locations per map; C69 frozen. Visibility and weak first-order power are computed candidates, not true mechanism labels.',stats=out)
(R/'audit/c72_continuation/c69_error_zones.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(out),flush=True)
