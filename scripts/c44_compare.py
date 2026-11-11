import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c44_dense_profile_20261102'
p=argparse.ArgumentParser();p.add_argument('--stage',default='fit64',choices=['fit64','fit256']);a=p.parse_args()
run=R/'runs'/f'c44_profile_{a.stage}';base=run/'fp32'
meta=json.loads((base/'dev64_full.json').read_text());pr=np.load(base/'dev64_full.npy',mmap_mode='r')
fit=json.loads((base/'fit_panel_full.json').read_text())
ref=R/'runs'/f'c43_path_{a.stage}'/'fp32'
rm=json.loads((ref/'dev64_full.json').read_text());rf=json.loads((ref/'fit_panel_full.json').read_text())
assert meta['indices']==rm['indices'] and fit['indices']==rf['indices']
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
ss=[];cc=[]
for j,i in enumerate(meta['indices']):
 s=[];c=[]
 for b in [0,1]:
  m=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool)
  e=np.asarray(pr[j,:,b]).reshape(256,256)[m].astype(float)-Y[b][i][m].astype(float)
  s.append(e@e);c.append(m.sum())
 ss.append(s);cc.append(c)
ss=np.array(ss);cc=np.array(cc);measured=np.sqrt(ss.sum(0)/cc.sum(0))
assert np.max(abs(measured-meta['RMSE']))<1e-9
np.savez(base/'independent_dev64.npz',ids=meta['indices'],sse=ss,count=cc)
control=np.load(ref/'independent_dev64.npz')
assert control['ids'].tolist()==meta['indices'] and np.array_equal(control['count'],cc)
tids=sorted({rows[i]['terrain_id'] for i in meta['indices']});s0=[];s1=[];ct=[]
for t in tids:
 ix=np.array([rows[i]['terrain_id']==t for i in meta['indices']])
 s0.append(control['sse'][ix].sum(0));s1.append(ss[ix].sum(0));ct.append(cc[ix].sum(0))
s0=np.array(s0);s1=np.array(s1);ct=np.array(ct)
ix=np.random.default_rng(20261102).integers(0,len(tids),(2000,len(tids)))
delta=np.sqrt(s1[ix].sum(1)/ct[ix].sum(1))-np.sqrt(s0[ix].sum(1)/ct[ix].sum(1))
gain=np.array(rm['RMSE'])-measured;fitgain=np.array(rf['RMSE'])-np.array(fit['RMSE'])
c0=json.loads((R/'runs'/f'c43_path_{a.stage}'/'config.json').read_text())
c1=json.loads((run/'config.json').read_text())
assert c0['order_sha256']==c1['order_sha256'] and c0['batch']==c1['batch'] and c0['steps']==c1['steps']
out={'stage':a.stage,'scope':'same frozen internalDEV64,264 full maps; TRAIN-only training',
 'C44_RMSE':measured.tolist(),'C43_path_RMSE':rm['RMSE'],'C44_FIT_panel_RMSE':fit['RMSE'],'C43_FIT_panel_RMSE':rf['RMSE'],
 'DEV_improvement_db':gain.tolist(),'FIT_improvement_db':fitgain.tolist(),
 'C44_minus_C43_CI95':np.quantile(delta,[.025,.975],axis=0).T.tolist(),
 'C38_same_DEV64':json.loads((A/'protocol.json').read_text())['C38_DEV64'],
 'independent_score_verified':True,'same_map_order_batch_steps':True,
 'screen64_pass':bool((gain>=.5).all() and (fitgain>=.5).all()) if a.stage=='fit64' else None,
 'warm_mean_seconds_both_maps':meta['warm_mean_seconds_both_maps'],
 'limits':'Module changes include dense profile, learned convolution and empirical phase candidate; cannot attribute gain to order alone. One seed. All scores internalDEV, not officialVAL.'}
(A/f'comparison_{a.stage}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
