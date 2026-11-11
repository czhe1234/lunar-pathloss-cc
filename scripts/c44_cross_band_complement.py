"""Check whether C44 contributes cross-band residual information beyond C38."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c44_dense_profile_20261102'
p=json.loads((A/'protocol.json').read_text());ids=p['dev64_ids']
pr=np.load(R/'runs/c44_profile_fit64/fp32/dev64_full.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
ref={}
for d in (R/'runs/c38_1024_continue/predictions').glob('train_dev_c34_fresh_shard*'):
 ix=json.loads((d/'indices.json').read_text())['index_rows'];v=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(ix):ref[i]=(v,j)
G=np.zeros((2,2,2));g=np.zeros((2,2));s=np.zeros(2);count=np.zeros(2)
mode={n:np.zeros(2) for n in ['C38','C44']};common_count=0
for j,i in enumerate(ids):
 arr,k=ref[i];base=np.stack([np.asarray(v[k]).reshape(-1).astype(float) for v in arr])
 new=np.asarray(pr[j]).T.astype(float);y=np.stack([np.asarray(v[i]).reshape(-1).astype(float) for v in Y])
 masks=np.stack([np.unpackbits(v[i],bitorder='big').astype(bool) for v in M])
 delta=new-base;err=base-y
 for b in [0,1]:
  x=delta[:,masks[b]];e=err[b,masks[b]]
  G[b]+=x@x.T;g[b]+=x@e;s[b]+=e@e;count[b]+=len(e)
 common=masks[0]&masks[1];common_count+=common.sum()
 for key,e in [('C38',err),('C44',new-y)]:
  modes=np.stack(((e[0]+e[1])/2,(e[1]-e[0])/2))[:,common]
  mode[key]+=(modes*modes).sum(1)
betas=[]
for b in [0,1]:
 lam=.01*np.trace(G[b])/2
 betas.append(np.clip(-np.linalg.solve(G[b]+lam*np.eye(2),g[b]),-.5,.5))
betas=np.array(betas)
blended=np.sqrt((s+2*np.einsum('bi,bi->b',betas,g)+np.einsum('bi,bij,bj->b',betas,G,betas))/count)
base=np.sqrt(s/count);gain=base-blended
out={'scope':'Posthoc DEV64 calibration only; not independent validation or a submission score',
 'formula':'PL_blend_band=PL_C38_band+beta_band_low*(C44low-C38low)+beta_band_high*(C44high-C38high)',
 'ridge':'1% of mean diagonal Gram; each beta clipped[-0.5,0.5], fixed before calculation',
 'beta':betas.tolist(),'C38_RMSE':base.tolist(),'calibration_RMSE':blended.tolist(),'calibration_gain_db':gain.tolist(),
 'common_valid_pixels':int(common_count),
 'common_mode_RMSE':{k:float(np.sqrt(v[0]/common_count)) for k,v in mode.items()},
 'band_difference_RMSE':{k:float(2*np.sqrt(v[1]/common_count)) for k,v in mode.items()},
 'confirm_remaining548_DEV_if_both_gain_at_least010db':bool((gain>=.1).all()),
 'official_VAL_loaded':False,'website_submission':False}
(A/'cross_band_complement64.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
