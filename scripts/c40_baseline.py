from pathlib import Path
import sys,csv,json
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R))
from c34_inference import Predictor
p=Predictor(R/'runs/c37_wide_periodic8')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
cfg=json.loads((R/'audit/c40_rt_pilot_20261101/map0_floor_128x4000000/config.json').read_text());i=cfg['index'];row=rows[i]
pr=p.predict(H[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']))
np.save(R/f'audit/c40_rt_pilot_20261101/C37_train_map{i}.npy',pr)
print(json.dumps({'index':i,'model':'C37 FIT only','shape':list(pr.shape)}),flush=True)
