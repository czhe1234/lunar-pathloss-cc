"""Identify fixed cell interpolation fingerprints across different TRAIN terrains."""
import csv,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c88_mesh';D=R.parent/'data';DIRS=[(0,1),(1,0),(1,1),(1,-1)]
def high(x):
    return x-uniform_filter(x.astype(np.float64),size=9,mode='nearest')
def shift(x,dy,dx):
    return np.roll(x,(-dy,-dx),(0,1))
def corr(a,b):
    a=high(a)[5:-5,5:-5].ravel();b=high(b)[5:-5,5:-5].ravel()
    return float(np.corrcoef(a,b)[0,1])
def signature(idx,w):
    idx=idx.reshape(256,256,3);w=w.reshape(256,256,3)
    out=[]
    for dy,dx in DIRS:
        ids2=shift(idx,dy,dx);w2=shift(w,dy,dx)
        cov=((idx[:,:,:,None]==ids2[:,:,None,:])*w[:,:,:,None]*w2[:,:,None,:]).sum((2,3))
        out.append(np.sqrt(np.maximum((w*w).sum(2)+(w2*w2).sum(2)-2*cov,0)))
    return np.stack(out)
def main():
    dev=set(json.load(open(R/'audit/internal_split.json'))['dev_terrain_ids'])
    rows=list(csv.DictReader(open(D/'train_index.csv')));pool=[i for i,r in enumerate(rows) if r['terrain_id'] not in dev and r['tx_index']=='0']
    rng=np.random.default_rng(881109);ids=rng.choice(pool,2048,replace=False)
    ys=[np.load(D/f'train_pl{s}.npy',mmap_mode='r') for s in ('415','58')]
    ms=[np.load(D/f'train_mask{s}.npy',mmap_mode='r') for s in ('415','58')]
    vals=np.zeros((2,2,4,256,256),np.float64);counts=np.zeros_like(vals);t=time.monotonic()
    for j,i in enumerate(ids):
        k=j%2
        for b in range(2):
            y=np.asarray(ys[b][i],dtype='float32');w=np.unpackbits(ms[b][i]).reshape(256,256).astype(bool)
            for z,(dy,dx) in enumerate(DIRS):
                ww=w&shift(w,dy,dx);v=np.minimum(abs(y-shift(y,dy,dx)),12.)*ww
                # Maps/patches with greater radio contrast should not dominate a fixed spatial pattern.
                val=v/(.5+uniform_filter(v,size=9,mode='nearest'));vals[k,b,z]+=val;counts[k,b,z]+=ww
        if (j+1)%512==0:print(j+1,round(time.monotonic()-t),flush=True)
    sig=(vals/counts.clip(min=1)).astype(np.float32)
    np.savez(A/'fingerprint_train.npz',ids=ids,signature=sig)
    reliability=np.array([[corr(sig[0,b,k],sig[1,b,k]) for k in range(4)] for b in range(2)])
    report={'maps_per_independent_half':1024,'split_half':reliability.tolist(),'solid_candidate':{}}
    for f in sorted((A/'tables').glob('*.npz')):
        v=np.load(f);s=signature(v['ids'],v['weights'])
        cc=np.array([[corr(sig[:,b,k].mean(0),s[k]) for k in range(4)] for b in range(2)])
        control=np.array([[corr(sig[:,b,k].mean(0),shift(s[k],37,59)) for k in range(4)] for b in range(2)])
        report['solid_candidate'][f.stem]={'match_corr':cc.tolist(),'rolled_corr':control.tolist(),'mean':float(cc.mean())}
    (A/'fingerprint.json').write_text(json.dumps(report,indent=2)+'\n')
    print('split-half',reliability.round(3),'correlations',{k:round(v['mean'],3) for k,v in report['solid_candidate'].items()},flush=True)
if __name__=='__main__':main()
