"""FIT-only candidate selection, paired validation on separate internal DEV terrains."""
import csv
import json
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'audit/c66_transport'
for folder in [ROOT/f'features/c66_frozen_maps/fit{s}of2' for s in [0,1]]:
    while not (folder/'complete.json').exists():
        time.sleep(2)
panel=np.load(A/'lattice_panel.npz')
indices=panel['maps']
pixels=panel['pixels']
rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()))
heights=np.load(ROOT.parent/'data/train_hm.npy',mmap_mode='r')
geos=np.load(ROOT/'features/c4_extra/train.npy',mmap_mode='r')
def prediction_sources():
    lookup={}
    for folder in sorted((ROOT/'features/c66_frozen_maps').glob('fit*of2')):
        arr=np.load(folder/'prediction.npy',mmap_mode='r')
        for j,i in enumerate(json.loads((folder/'indices.json').read_text())):
            lookup[i]=(arr,j)
    folder=ROOT/'runs/c54_combo_dev/bf16_compiled/all_shard0of1'
    values=[np.load(folder/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
    for j,i in enumerate(json.loads((folder/'indices.json').read_text())['index_rows']):
        lookup[i]=(values,j)
    return lookup
sources=prediction_sources()
base=[];gap=[]
for i,p in zip(indices,pixels):
    src,j=sources[int(i)]
    pred=src[j] if isinstance(src,np.ndarray) else np.stack([v[j] for v in src])
    base.append(pred.reshape(2,-1)[:,p].T)
    row=rows[i];h=heights[int(row['hm_row']),int(row['tx_row']),int(row['tx_col'])]
    critical=2.5+2*np.sinh(3*geos[i,0].ravel()[p].astype('float32'))
    gap.append(critical-(3-np.remainder(h,1)))
base=np.array(base);gap=np.array(gap)
targets=[];masks=[]
for b in ['415','58']:
    y=np.load(ROOT.parent/'data'/f'train_pl{b}.npy',mmap_mode='r')
    m=np.load(ROOT.parent/'data'/f'train_mask{b}.npy',mmap_mode='r')
    targets.append(np.array([y[i].ravel()[p] for i,p in zip(indices,pixels)]))
    masks.append(np.array([np.unpackbits(m[i])[p] for i,p in zip(indices,pixels)],bool))
target=np.stack(targets,-1).astype('float64');mask=np.stack(masks,-1)
error=target-base
feat=np.load(A/'lattice_features.npy',mmap_mode='r')
def rmse(errors, valid):
    return np.sqrt((errors**2*valid).sum((0,1))/valid.sum((0,1)))
ref=rmse(error[64:],mask[64:])
records=[]
for phys in range(2):
 for diag in range(2):
  for k,n in enumerate(panel['rays']):
    val=feat[phys,diag,k]
    correction=-10*np.log10((val[...,0]+.03)/1.03)
    dblog=-10*val[...,1]
    expected=np.log1p(np.clip(val[...,2],0,100))
    def features(sel,actual):
        v=[np.ones_like(expected[sel]),expected[sel],expected[sel]**2]
        if actual:v.extend([correction[sel],dblog[sel]])
        return np.stack(v,-1)
    reports={}
    for actual,name in [(False,'geometric_count_control'),(True,'lattice')]:
      pr=np.zeros_like(error)
      for zone in [gap<-.4, (gap>=-.4)&(gap<=.4), gap>.4]:
       for b in range(2):
        cal=zone[:64]&mask[:64,:,b]
        z=features(slice(0,64),actual)[cal].astype(float)
        beta=np.linalg.solve(z.T@z+np.eye(z.shape[1])*3,z.T@error[:64,:,b][cal])
        design=features(slice(None),actual)
        pr[:,:,b]+=zone*np.clip(design@beta,-15,15)
      reports[name]={'FIT':rmse((error-pr)[:64],mask[:64]).tolist(),'DEV':rmse((error-pr)[64:],mask[64:]).tolist()}
    records.append({'grid':phys,'diagonal':diag,'rays':int(n),**reports})
best=min(records,key=lambda o:np.mean(o['lattice']['FIT']))
out={'baseline_DEV':ref.tolist(),'selection':'geometry, direction count and correction coefficients chosen only on 64 FIT terrains. Final report on 64 different DEV terrains.','selected':best,'gain':(ref-best['lattice']['DEV']).tolist(),'lattice_gain_over_expected_count_control':(np.array(best['geometric_count_control']['DEV'])-best['lattice']['DEV']).tolist(),'records':records}
(A/'lattice_screen.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
