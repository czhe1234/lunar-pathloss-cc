"""Diagnostic two-scalar blend upper bound on DEV64; not a submission score."""
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
 indices=json.loads((d/'indices.json').read_text())['index_rows']
 pred=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(indices):ref[i]=(pred,j)
s0=np.zeros(2);s1=np.zeros(2);ed=np.zeros(2);dd=np.zeros(2);cross=np.zeros(2);count=np.zeros(2)
for j,i in enumerate(ids):
 arrays,k=ref[i]
 for b in [0,1]:
  valid=np.unpackbits(M[b][i],bitorder='big').astype(bool)
  y=np.asarray(Y[b][i]).reshape(-1)[valid].astype(float)
  base=np.asarray(arrays[b][k]).reshape(-1)[valid].astype(float)
  new=np.asarray(pr[j,:,b])[valid].astype(float)
  e0=base-y;e1=new-y;delta=new-base
  s0[b]+=e0@e0;s1[b]+=e1@e1;ed[b]+=e0@delta;dd[b]+=delta@delta;cross[b]+=e0@e1;count[b]+=len(y)
w=np.clip(-ed/dd,-.5,.5);blended=np.sqrt((s0+2*w*ed+w*w*dd)/count)
base=np.sqrt(s0/count);gain=base-blended
out={'scope':'posthoc diagnostic on DEV64 used for C44 selection; coefficients fitted and scored on same panel, NOT independent validation',
     'formula':'PLblend=(1-w_band)*C38+w_band*C44; w constrained[-0.5,0.5]',
     'weights':w.tolist(),'C38_RMSE':base.tolist(),'C44_RMSE':np.sqrt(s1/count).tolist(),'calibration_blend_RMSE':blended.tolist(),
     'calibration_gain_db':gain.tolist(),'error_correlation':(cross/np.sqrt(s0*s1)).tolist(),
     'confirm_remaining548_DEV_only_if_both_gain_at_least_010db':bool((gain>=.1).all()),
     'no_new_model_training':True,'official_VAL_loaded':False,'website_submission':False,
     'interpretation':'If this best in-panel linear gain is small, this particular simple fusion lacks strong promotion evidence. This does not rule out nonlinear fusion or other architectures.'}
(A/'linear_complement64.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
