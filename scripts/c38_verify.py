"""Verify the actual exported C38 package against already completed VAL predictions."""
from pathlib import Path
import sys,time,json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];P=R/'releases/C38_20261102'
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'src'))
from c34_inference import Predictor
import c34_inference
assert Path(c34_inference.__file__).resolve().is_relative_to(P)
predictor=Predictor(P/'model')
rows=list(csv.DictReader((R.parent/'data/val_index.csv').open()))
H=np.load(R.parent/'data/val_hm.npy',mmap_mode='r')
d=R/'runs/c38_full_refit/predictions/val_full_c34_fresh_shard1of3'
pred=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
records=[]
for i in [1,64,127,511]:
 row=rows[i];t=time.time()
 q=predictor.predict(H[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']))
 ref=np.stack([x[i//3] for x in pred]);err=q-ref
 rec={'index':i,'shape':list(q.shape),'seconds':time.time()-t,'max_abs_difference_db':float(abs(err).max()),'RMSE_difference_db':np.sqrt(np.mean(err.astype(float)**2,axis=(1,2))).tolist()}
 assert np.isfinite(q).all() and np.max(abs(err))<.001,rec
 records.append(rec);print(json.dumps(rec),flush=True)
o=R/'audit/c38_package_verification.json'
o.write_text(json.dumps({'verified':True,'package':str(P),'GPU':predictor.device,'records':records,'input_contract':'One256x256heightmap in metres and integer TX row/col; no index, label, mask or feature-cache access inside Predictor','reference':'Same full-TRAIN C38 weights; raw-input batch VAL exporter'},indent=2)+'\n')
