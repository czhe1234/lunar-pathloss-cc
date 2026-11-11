"""Score completed C34/C35 input-only prediction shards with public masked RMSE."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,csv,time
import numpy as np
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',default='train',choices=['train','val']);p.add_argument('--shards',type=int,default=3);a=p.parse_args()
rows=list(csv.DictReader((R.parent/'data'/f'{a.split}_index.csv').open()))
if a.split=='train':
 dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
 expected=[i for i,r in enumerate(rows) if r['terrain_id'] in dev];part='dev'
else:expected=list(range(len(rows)));part='full'
Y=[np.load(R.parent/'data'/f'{a.split}_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'{a.split}_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
ids=[];ss=[];cc=[]
base=R/'runs'/a.run/'predictions'
for shard in range(a.shards):
 d=base/f'{a.split}_{part}_c34_fresh_shard{shard}of{a.shards}'
 assert (d/'inference.json').exists(),d
 index=json.loads((d/'indices.json').read_text())['index_rows'];ids.extend(index)
 pred=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(index):
  sums=[];counts=[]
  for k in [0,1]:
   valid=np.unpackbits(M[k][i],bitorder='big').reshape(256,256).astype(bool)
   q=np.asarray(pred[k][j]);assert np.isfinite(q).all()
   error=q[valid].astype(np.float64)-Y[k][i][valid].astype(np.float64)
   sums.append(float(error@error));counts.append(int(valid.sum()))
  ss.append(sums);cc.append(counts)
assert len(ids)==len(set(ids)) and set(ids)==set(expected)
ss=np.array(ss);cc=np.array(cc);rmse=np.sqrt(ss.sum(0)/cc.sum(0))
out={'scope':('complete internal DEV' if a.split=='train' else 'official fixed validation'),
 'run':a.run,'split':a.split,'maps':len(ids),'terrains':len({rows[i]['terrain_id'] for i in ids}),
 'RMSE':rmse.tolist(),'combined_rmse_db_masked':float(rmse.mean()),
 'valid_pixels':cc.sum(0).tolist(),'SSE':ss.sum(0).tolist(),'complete_requested_split':True,
 'formula':'Per band sqrt(sum(valid*(prediction-label)^2)/sum(valid)); combined arithmetic mean of two band RMSE',
 'website_submission':False}
dest=R/'runs'/a.run/f'score_{a.split}_complete.json'
dest.write_text(json.dumps(out,indent=2)+'\n')
np.savez(R/'runs'/a.run/f'per_map_{a.split}.npz',ids=ids,sse=ss,count=cc)
print(json.dumps(out),flush=True)
