from pathlib import Path
import argparse,csv,json,time
import numpy as np,torch
from c43_inference import Predictor,R
from cc_core import atomic_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
run=R/'runs'/a.run
metadata=json.loads((run/'dev64_full.json').read_text())
saved=np.load(run/'dev64_full.npy',mmap_mode='r')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
pr=Predictor(run/'best.pt')
torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
records=[]
for j in [0,64,127,255]:
    i=metadata['indices'][j];r=rows[i];h=np.array(H[int(r['hm_row'])],np.float32)
    torch.cuda.synchronize();start=time.perf_counter()
    pred=pr.predict(h,int(r['tx_row']),int(r['tx_col']))
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    ref=np.asarray(saved[j]).T.reshape(2,256,256)
    diff=pred.astype(float)-ref.astype(float)
    rms=float(np.sqrt((diff*diff).mean()))
    assert rms<.05,('inference interface differs from saved batch predictions',rms)
    records.append({'index':i,'seconds':seconds,'prediction_RMSE_difference_db':rms,'max_abs_difference_db':float(abs(diff).max())})
out={'run':a.run,'gpu':pr.gpu,'records':records,'warm_mean_seconds':float(np.mean([r['seconds'] for r in records[1:]])),
     'scope':'one raw input heightmap plus TX -> both256x256 PL maps, shared server H100; model initialization and disk IO excluded; no labels/masks read'}
atomic_json(run/'latency_interface.json',out);print(json.dumps(out),flush=True)
