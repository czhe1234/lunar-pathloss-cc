import sys,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c10_inference import C10Predictor
run=R/'runs/c10_points_phase_geometry_s20261101';model=C10Predictor(run);rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
ref=run/'predictions/train_dev_gpu_c10_fast_fresh_first4';ids=json.loads((ref/'indices.json').read_text())['index_rows'];P=[np.load(ref/f'pl{b}.npy') for b in ['415','58']]
results=[]
for j,i in enumerate(ids):
 row=rows[i];pr=model.predict(H[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']));expected=np.stack([p[j] for p in P]);d=float(abs(pr-expected).max());assert d==0
 results.append({'index_row':i,'max_abs':d,'float32_identical':True})
res={'standalone_input_interface':'Heightmap + TX pixel only','full_maps':len(ids),'results':results,'GPU':model.device,'passed':True}
(R/'audit/c10_standalone_verification.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)

