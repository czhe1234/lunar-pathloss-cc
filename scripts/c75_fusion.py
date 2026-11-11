"""Re-evaluate the prespecified mixture using the wider correction only after it passes its own gate."""
import csv,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def main():
    root=R/'runs/c75_wide_observed/full_internal';graph=R/'runs/c73_wide_real/full_internal';base=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
    ids=json.load(open(root/'indices.json'));assert ids==json.load(open(graph/'indices.json'))
    local=np.load(root/'correction.npy',mmap_mode='r');gp=np.load(graph/'prediction.npy',mmap_mode='r')
    bp=[np.load(base/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
    yy=[np.load(R.parent/f'data/train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ww=[np.load(R.parent/f'data/train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
    lookup={}
    for name in ('c73_compact','c73_dev_extra_compact'):
        for d in (R/'features'/name).glob('part*'):
            feat=np.load(d/'features.npy',mmap_mode='r');lookup.update({i:(feat,j) for j,i in enumerate(json.load(open(d/'indices.json')))})
    stats=np.zeros((len(ids),2,4,5))
    for j,i in enumerate(ids):
        phi,pos=lookup[i];vis=phi[pos,16].reshape(-1)
        for b in range(2):
            m=np.unpackbits(ww[b][i]).astype(bool);z=np.where(vis>.99,0,np.where(vis>=.01,1,np.where(phi[pos,41+b].reshape(-1)<-.8,3,2)))[m]
            y=yy[b][i].reshape(-1)[m].astype(np.float64);bs=bp[b][j].reshape(-1)[m].astype(np.float64)
            l=bs+local[j,b].reshape(-1)[m];g=gp[j,b].reshape(-1)[m].astype(np.float64)
            for k,v in enumerate(((bs-y)**2,(l-y)**2,(g-y)**2,(l-g)**2,np.ones_like(y))):
                stats[j,b,:,k]=np.bincount(z,weights=v,minlength=4)
    output=root/'fusion';output.mkdir(exist_ok=False)
    shuffle=np.load(graph/'sse_by_zone.npz')['shuffled'];np.savez(output/'sse_by_zone.npz',statistics=stats,shuffled=shuffle)
    sys.path.insert(0,str(R))
    from src.c74_mixture_score import summarize
    summarize(ids,stats,shuffle,output,local_name='C75')
if __name__=='__main__':main()
