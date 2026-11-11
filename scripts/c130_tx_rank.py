"""Leave-one-TX spatial-subspace diagnostic; query labels only for oracle projection."""
from pathlib import Path
import csv,json,time
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c130_projective';start=time.monotonic()
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
p=json.loads((R/'audit/c127_joint/protocol.json').read_text())
g=defaultdict(list);allg=defaultdict(list)
for i in p['fit']:g[rows[i]['terrain_id']].append(i)
for i,r in enumerate(rows):allg[r['terrain_id']].append(i)
terrains=sorted([t for t,ids in g.items() if len(ids)==4],key=int)
rng=np.random.default_rng(130041);terrains=[terrains[i] for i in rng.permutation(len(terrains))[:32]]
height=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
ys=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ms=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
rr,cc=np.indices((256,256));ks=[0,1,2,4,8,16,32,49]
totals=np.zeros((2,2,len(ks),2));records=[];cover=[];nearest_stats=np.zeros((2,2,2))
for t in terrains:
 ids=allg[t];assert len(ids)==51
 h=np.asarray(height[int(rows[ids[0]]['hm_row'])],dtype='float64')
 source=np.array([[int(rows[i]['tx_row']),int(rows[i]['tx_col'])] for i in ids])
 zz=h[source[:,0],source[:,1]]+3
 dist=np.sqrt((rr-source[:,0,None,None])**2+(cc-source[:,1,None,None])**2+(h+1-zz[:,None,None])**2).clip(.25)
 fs=20*np.log10(dist)+20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458.)[:,None,None,None]
 terrain_rec=np.zeros_like(totals)
 for b in range(2):
  valid=np.unpackbits(ms[b][ids],axis=1).reshape(51,256,256).astype(bool)
  common=valid.all(0);cover.append([int(t),b,int(common.sum()),int(valid.sum())])
  if common.sum()<2048:continue
  y=np.asarray(ys[b][ids],dtype='float64')
  excess=y-fs[b]
  # Filtering uses normalized valid support. Oracle highpass is diagnostic, never model input.
  denom=gaussian_filter(valid.astype('float64'),sigma=(0,4,4),mode='nearest')
  low=gaussian_filter(np.where(valid,excess,0),sigma=(0,4,4),mode='nearest')/np.maximum(denom,1e-8)
  high=excess-low
  for kind,field in enumerate([excess,high]):
   X=field[:,common];G=X@X.T;n=X.shape[1]
   for q in range(51):
    ix=np.delete(np.arange(51),q);K=G[np.ix_(ix,ix)];m=K.mean(0);grand=K.mean()
    centered=K-m[:,None]-m[None,:]+grand
    cross=G[q,ix]-G[q,ix].mean()-m+grand
    qnorm=max(0.,G[q,q]-2*G[q,ix].mean()+grand)
    eigen,U=np.linalg.eigh(centered);eigen=eigen[::-1];U=U[:,::-1]
    gain=np.divide((U.T@cross)**2,eigen,out=np.zeros(50),where=eigen>max(eigen[0]*1e-10,1e-10))
    prefix=np.r_[0,np.cumsum(gain)]
    for k,rank in enumerate(ks):terrain_rec[kind,b,k]+=[max(0,qnorm-prefix[rank]),n]
    near=ix[np.argmin(((source[ix]-source[q])**2).sum(1))]
    s=G[q,q]+G[near,near]-2*G[q,near]
    nearest_stats[kind,b]+=[s,n]
 totals+=terrain_rec;records.append(terrain_rec)
 progress=dict(terrains=len(records),total=len(terrains),seconds=time.monotonic()-start)
 (A/'rank_progress.json').write_text(json.dumps(progress));print(progress,flush=True)
out=dict(status='complete',terrains=terrains,ranks=ks,RMSE=np.sqrt(totals[:,:,:,0]/totals[:,:,:,1]).tolist(),
 kinds=['excess_dB_over_FS','fine_excess_sigma4'],bands=['415MHz','5.775GHz'],
 nearest_TX_RMSE=np.sqrt(nearest_stats[:,:,0]/nearest_stats[:,:,1]).tolist(),coverage=cover,seconds=time.monotonic()-start,
 scope='32FITterrains,51TX each; common-valid support within each terrain/band; leave-query-TX out before centering and basis estimation. Coefficients of query projection use its labels: privileged diagnostic, not deployable. Rank0 is other50TX mean; rank49 is all centered donor span.',official_val_used=False,train_launched=False)
np.savez(A/'rank_stats.npz',per_terrain=records,terrains=np.array(terrains),totals=totals)
(A/'tx_rank_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
