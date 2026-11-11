"""Cluster bootstrap and independent SVD check of the projection algebra."""
from pathlib import Path
import json,csv,numpy as np
from scipy.ndimage import gaussian_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c130_projective'
z=np.load(A/'projective_stats.npz');stats=z['stats'];rng=np.random.default_rng(130601)
draw=rng.integers(len(stats),size=(1500,len(stats)))
ci={}
for bk,name in enumerate(['all_common','near_direction_match']):
 s=stats[draw,bk,2].sum(1);n=s[:,:,0];cov=s[:,:,5]/n-(s[:,:,1]/n)*(s[:,:,2]/n)
 variance=(s[:,:,3]/n-(s[:,:,1]/n)**2)*(s[:,:,4]/n-(s[:,:,2]/n)**2)
 corr=cov/np.sqrt(variance);ci[name]=np.quantile(corr,[.025,.975],axis=0).T.tolist()
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
rank=json.loads((A/'tx_rank_result.json').read_text());t=rank['terrains'][0]
ids=[i for i,r in enumerate(rows) if r['terrain_id']==str(t)];assert len(ids)==51
h=np.array(np.load(R.parent/'data/train_hm.npy',mmap_mode='r')[int(rows[ids[0]]['hm_row'])],dtype='float64')
rr,cc=np.indices((256,256));source=np.array([[int(rows[i]['tx_row']),int(rows[i]['tx_col'])] for i in ids])
dist=np.sqrt((rr-source[:,0,None,None])**2+(cc-source[:,1,None,None])**2+(h+1-h[source[:,0],source[:,1]][:,None,None]-3)**2).clip(.25)
fs=20*np.log10(dist)+20*np.log10(4*np.pi*415e6/299792458.)
Y=np.array(np.load(R.parent/'data/train_pl415.npy',mmap_mode='r')[ids],dtype='float64')
v=np.unpackbits(np.load(R.parent/'data/train_mask415.npy',mmap_mode='r')[ids],axis=1).reshape(51,256,256).astype(bool)
excess=Y-fs;low=gaussian_filter(np.where(v,excess,0),sigma=(0,4,4),mode='nearest')/gaussian_filter(v.astype('float64'),sigma=(0,4,4),mode='nearest').clip(1e-8)
X=(excess-low)[:,v.all(0)];mean=X[1:].mean(0);B=X[1:]-mean;query=X[0]-mean
beta=np.linalg.lstsq(B.T,query,rcond=1e-5)[0];direct=float(np.sum((query-beta@B)**2))
K=B@B.T;g=B@query;e,U=np.linalg.eigh(K);keep=e>max(e[-1]*1e-10,1e-10)
via_gram=float(query@query-np.sum((U[:,keep].T@g)**2/e[keep]))
assert abs(direct-via_gram)<1e-6*max(direct,1)
out=dict(status='passed',source_direction_corr_terrain_CI95=ci,rank_independent_real_case=dict(terrain=int(t),query_map=ids[0],pixels=X.shape[1],direct_lstsq_SSE=direct,Gram_SSE=via_gram),
 numerical_scope='one real terrain/query via direct least-squares versus Gram eigenvalue algebra; not a new model score')
(A/'diagnostics_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
