"""CPU-only, independent float64 C54 vs C38 saved-prediction scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,csv,json
import numpy as np
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scope',choices=['dev64','remaining','all'],default='dev64');a=p.parse_args()
allids=np.load(R/'features/c6_points/dev_map_indices.npy').tolist()
small=json.loads((R/'audit/c43_architecture_20261102/protocol.json').read_text())['dev64_ids']
expected=set(small if a.scope=='dev64' else (allids if a.scope=='all' else [i for i in allids if i not in set(small)]))
folders=sorted((R/'runs'/a.run/'fp32').glob(a.scope+'_shard*'))
if a.scope=='all':folders=sorted((R/'runs'/a.run/'fp32').glob('dev64_shard*'))+sorted((R/'runs'/a.run/'fp32').glob('remaining_shard*'))
new={}
for folder in folders:
 meta=json.loads((folder/'inference.json').read_text());assert meta['complete'] and meta['no_target_or_validity_mask_read']
 arrays=[np.load(folder/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(json.loads((folder/'indices.json').read_text())['index_rows']):
  assert i not in new;new[i]=(arrays,j)
assert set(new)==expected,(len(new),len(expected))
old={}
for folder in sorted((R/'runs/c38_1024_continue/predictions').glob('train_dev_c34_fresh_shard*')):
 arrays=[np.load(folder/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(json.loads((folder/'indices.json').read_text())['index_rows']):old[i]=(arrays,j)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
by={};per=[];newss=[];oldss=[];counts=[]
for i in sorted(expected):
 sn=[];so=[];cc=[]
 for b in range(2):
  keep=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool)
  n,j=new[i];o,k=old[i];target=Y[b][i][keep].astype(np.float64)
  en=n[b][j][keep].astype(np.float64)-target;eo=o[b][k][keep].astype(np.float64)-target
  assert np.isfinite(en).all();sn.append(float(en@en));so.append(float(eo@eo));cc.append(int(keep.sum()))
 newss.append(sn);oldss.append(so);counts.append(cc)
 t=rows[i]['terrain_id'];acc=by.setdefault(t,[np.zeros(2),np.zeros(2),np.zeros(2)])
 acc[0]+=sn;acc[1]+=so;acc[2]+=cc
 per.append({'index':i,'terrain':t,'C54_SSE':sn,'C38_SSE':so,'count':cc})
ns=np.array(newss).sum(0);os_=np.array(oldss).sum(0);cs=np.array(counts).sum(0)
nr=np.sqrt(ns/cs);orr=np.sqrt(os_/cs)
groups=[by[t] for t in sorted(by)];sn=np.array([v[0] for v in groups]);so=np.array([v[1] for v in groups]);cc=np.array([v[2] for v in groups])
ix=np.random.default_rng(20261102).integers(0,len(groups),(2000,len(groups)))
delta=np.sqrt(sn[ix].sum(1)/cc[ix].sum(1))-np.sqrt(so[ix].sum(1)/cc[ix].sum(1))
out={'scope':a.scope,'precision':'FP32_correction; unchanged4096BF16teacher','maps':len(expected),'terrains':len(groups),
 'C54_RMSE':nr.tolist(),'C38_RMSE':orr.tolist(),'gain_db':(orr-nr).tolist(),'combined':float(nr.mean()),'count':cs.tolist(),
 'delta_C54_minus_C38_CI95':np.quantile(delta,[.025,.975],axis=0).T.tolist(),
 'both_gain010db':bool(((orr-nr)>=.10).all()),'both_le_1db':bool((nr<=1).all()),
 'independent_float64_masked_scoring':True,'official_VAL':False,'limits':'One seed. Terrain bootstrap conditions on trained model; internal DEV used for selection.'}
folder=R/'audit/c54_band_experts_20261102';(folder/(a.run+'_'+a.scope+'_score.json')).write_text(json.dumps(out,indent=2)+'\n')
np.savez(folder/(a.run+'_'+a.scope+'_permap.npz'),indices=sorted(expected),new_sse=newss,old_sse=oldss,count=counts)
print(json.dumps(out,indent=2),flush=True)
