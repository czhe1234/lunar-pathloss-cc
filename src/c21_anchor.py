import json,csv,time
from pathlib import Path
import numpy as np
from c21_data import R,D
p=R/'runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh';ids=json.loads((p/'indices.json').read_text())['index_rows'];rows=list(csv.DictReader((D/'train_index.csv').open()))
out={'scope':'complete2562 terrain-disjoint DEV maps; frozenFIT-only C10 predictions','source':str(p),'bands':{},'per_map':[]}
ss=np.zeros((len(ids),2));cc=np.zeros_like(ss)
for b,band in enumerate(['415','58']):
 pred=np.load(p/f'pl{band}.npy',mmap_mode='r');y=np.load(D/f'train_pl{band}.npy',mmap_mode='r');m=np.load(D/f'train_mask{band}.npy',mmap_mode='r')
 for j,i in enumerate(ids):
  mask=np.unpackbits(m[i],bitorder='big').reshape(256,256).astype(bool);e=pred[j].astype(np.float64)-np.asarray(y[i],np.float64);ss[j,b]=np.square(e)[mask].sum();cc[j,b]=mask.sum()
 out['bands'][band]={'rmse':float(np.sqrt(ss[:,b].sum()/cc[:,b].sum())),'sse':float(ss[:,b].sum()),'count':int(cc[:,b].sum())}
out['per_map']=[{'id':i,'terrain_id':rows[i]['terrain_id'],'sse':s.tolist(),'count':c.tolist()} for i,s,c in zip(ids,ss,cc)]
out['score']=float(np.sqrt(ss.sum(0)/cc.sum(0)).mean())
a=R/'audit/c21_execution_20261101';(a/'C10_anchor.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='per_map'}))
