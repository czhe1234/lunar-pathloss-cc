"""Check whether published terrain IDs recover 4x4 parent adjacency using H only."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment'
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));mapping={int(r['terrain_id']):int(r['hm_row']) for r in rows}
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');rng=np.random.default_rng(951024)
records=[]
for rowmajor in (True,False):
    dx,dy=(1,4) if rowmajor else (4,1);out={}
    for axis,delta in [('x',dx),('y',dy)]:
        pairs=[]
        for i in sorted(mapping):
            coord=i%4 if (rowmajor==(axis=='x')) else (i//4)%4
            j=i+delta
            if coord<3 and j in mapping and i//16==j//16:pairs.append((i,j))
        signal=[];wrong=[];unaligned=[]
        for i,j in pairs:
            h=H[mapping[i]];g=H[mapping[j]]
            a,b=(h[:,-1],g[:,0]) if axis=='x' else (h[-1,:],g[0,:])
            r=a-b;signal.append(float(np.std(r)))
            k=int(rng.choice(list(mapping)));g=H[mapping[k]];bb=g[:,0] if axis=='x' else g[0,:]
            rr=a-bb;wrong.append(float(np.std(rr)))
            if len(signal)<5:unaligned.append([i,j,float(np.mean(r)),float(np.std(r))])
        out[axis]=dict(pairs=len(pairs),matched_median=float(np.median(signal)),random_median=float(np.median(wrong)),
                      matched_quantiles=np.quantile(signal,[0,.1,.5,.9,1]).tolist(),examples=unaligned)
    records.append(dict(rowmajor=rowmajor,bands=out))
(A/'parent_edge_test.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
