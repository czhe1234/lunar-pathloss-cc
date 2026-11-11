"""Use only development-terrain validity geometry to constrain export support; no power labels."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss';D=R.parent/'data'
plan=json.loads((A/'plan.json').read_text());ids=[e['index'] for e in plan['screen']+plan['extension']]
assert len(ids)==60
summary={}
for b in ('415','58'):
    w=np.load(D/f'train_mask{b}.npy',mmap_mode='r');v=np.stack([np.unpackbits(w[i]).reshape(256,256) for i in ids]);n=v.sum(0)
    xy=np.stack(np.where(n==0),1);summary[b]=dict(maps=len(ids),always_invalid=xy.tolist(),always_valid=int((n==len(ids)).sum()),
       corner_valid=[int(n[r,c]) for r,c in [(0,0),(0,255),(255,0),(255,255)]],
       edge_valid_fraction=[float(x.mean()) for x in [v[:,0,:],v[:,-1,:],v[:,:,0],v[:,:,-1]]])
# Exact candidate support without any propagation assumptions.
for name in ['numpy_mean_default','sionna_centers_default']:
    z=np.load(R/'audit/c94_measurement/templates_numpy22'/f'{name}.npz')
    # Stored templates replace unsupported corners; recompute support from centers independently.
    from scipy.spatial import Delaunay
    if name.startswith('sionna'):cent=np.load(R/'audit/c94_measurement/sionna_centers.npy')
    else:
        x=np.linspace(-128,128,256,dtype='float32');xx,yy=np.meshgrid(x,-x);v=np.stack([xx.ravel(),yy.ravel()],1)
        c=np.arange(65536).reshape(256,256)[:-1,:-1].ravel();f=np.concatenate([np.stack([c,c+1,c+257],1),np.stack([c,c+257,c+256],1)])
        cent=v[f].mean(1)
    tri=Delaunay(cent);ss=tri.find_simplex(z['query'])
    summary[name]=dict(outside=np.stack(np.where((ss<0).reshape(256,256)),1).tolist())
(A/'mask_export_support.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary)[:5000])
