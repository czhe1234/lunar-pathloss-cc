"""Terrain-only intrinsic dimension and template check. No PL or validity file opened."""
import csv,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from types import SimpleNamespace
R=Path(__file__).resolve().parents[1];A=R/'audit/c103_terrain_structure';A.mkdir(exist_ok=True)
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));h=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
fitmap=np.load(R/'features/c38_dense/fit_map_indices.npy');devmap=np.load(R/'features/c6_points/dev_map_indices.npy')
fit=np.unique([int(rows[int(i)]['hm_row']) for i in fitmap]);dev=np.unique([int(rows[int(i)]['hm_row']) for i in devmap])
assert len(fit)==5508 and len(dev)==612 and not set(fit)&set(dev)
hm=np.array(h);small=hm.reshape(len(hm),32,8,32,8).mean((2,4))
mean=small.mean((1,2));std=small.std((1,2)).clip(.0001);shape=(small-mean[:,None,None])/std[:,None,None]
x=shape[fit].reshape(len(fit),-1).astype('float64');t=time.monotonic()
centre=x.mean(0)
_,singular,vt=svd(x-centre,full_matrices=False,lapack_driver='gesdd')
pca=SimpleNamespace(components_=vt[:64],mean_=centre,explained_variance_ratio_=singular[:64]**2/(singular**2).sum())
codes=(shape.reshape(len(hm),-1)-centre)@pca.components_.T;fitnorm=(x/32).astype('float32')
best=np.full(len(dev),np.inf);match=np.full(len(dev),-1);transform=np.full(len(dev),-1)
for k in range(8):
 a=shape[dev]
 if k>=4:a=a[:,:,::-1]
 a=np.rot90(a,k%4,axes=(1,2)).reshape(len(dev),-1).copy()/32
 dot=a@fitnorm.T
 d2=np.maximum(0,(a*a).sum(1)[:,None]+(fitnorm*fitnorm).sum(1)[None]-2*dot);where=d2.argmin(1)
 dist=np.sqrt(((a.astype('float64')-fitnorm[where].astype('float64'))**2).sum(1))
 take=dist<best;best[take]=dist[take];match[take]=fit[where[take]];transform[take]=k
out=dict(fit_terrains=len(fit),dev_terrains=len(dev),pooled_grid=32,normalization='per terrain subtractmean and divide elevationstd; affine-height template check only',
 explained_cumulative={str(k):float(pca.explained_variance_ratio_[:k].sum()) for k in [4,8,16,32,64]},
 dev_nearest_affineD4_relative_RMS_quantiles=np.quantile(best,[0,.01,.1,.5,.9,1]).tolist(),
 near_template_counts={str(th):int((best<th).sum()) for th in [1e-5,.001,.01,.05,.1]},
 source='Only officialTRAIN H andTXindex metadata. PCA fit5508FIT; transform612DEV. No labels or masks opened.',
 seconds=time.monotonic()-t)
np.savez(A/'terrain_codes.npz',hm_row=np.arange(len(hm)),mean=mean,std=std,codes=codes.astype('float32'),components=pca.components_.astype('float32'),pca_mean=pca.mean_.astype('float32'),fit=fit,dev=dev,nearest_match=match,nearest_distance=best,nearest_transform=transform)
(A/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
