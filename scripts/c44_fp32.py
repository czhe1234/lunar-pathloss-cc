"""Canonical FP32 input-only replay of selected C43 checkpoints on TRAIN-derived panels."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard,atomic_json
from c44_inference import Predictor
import argparse,json,csv,time,hashlib
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
run=R/'runs'/a.run;assert a.run.startswith('c44_')
res=json.loads((run/'result.json').read_text());out=run/'fp32';out.mkdir(exist_ok=False)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
pr=Predictor(run/'best.pt',amp=False);torch.backends.cudnn.benchmark=True
for panel in ['dev64_full','fit_panel_full']:
    ids=json.loads((run/f'{panel}.json').read_text())['indices']
    dest=np.lib.format.open_memmap(out/f'{panel}.npy',mode='w+',dtype=np.float32,shape=(len(ids),65536,2))
    per=[];ss=np.zeros(2);cc=np.zeros(2);times=[]
    for j,i in enumerate(ids):
        row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32)
        torch.cuda.synchronize();t=time.perf_counter()
        pred=pr.predict(h,int(row['tx_row']),int(row['tx_col']))
        torch.cuda.synchronize();times.append(time.perf_counter()-t)
        dest[j]=pred.reshape(2,-1).T
        sums=[];counts=[]
        for b in range(2):
            valid=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool)
            e=pred[b][valid].astype(np.float64)-Y[b][i][valid].astype(np.float64)
            sums.append(float(e@e));counts.append(int(valid.sum()))
        ss+=sums;cc+=counts
        per.append({'index':i,'terrain':row['terrain_id'],'SSE':sums,'count':counts})
        if (j+1)%64==0:print(json.dumps({'run':a.run,'panel':panel,'maps':j+1}),flush=True)
    dest.flush();rm=np.sqrt(ss/cc)
    record={'RMSE':rm.tolist(),'combined':float(rm.mean()),'SSE':ss.tolist(),'count':cc.tolist(),'maps':len(ids),'pixels_per_map':65536,
            'indices':ids,'per_map':per,'precision':'FP32, TF32 disabled','inference_inputs':'raw heightmap and TX only',
            'warm_mean_seconds_both_maps':float(np.mean(times[1:])),
            'warm_p95_seconds_both_maps':float(np.quantile(times[1:],.95)),
            'first_seconds':times[0],'gpu':pr.gpu,
            'weights_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),
            'selection':'Unchanged checkpoint selected by predeclared BF16 monitor; no reselection during FP32 replay',
            'official_VAL_loaded':False}
    atomic_json(out/f'{panel}.json',record);print(json.dumps({k:record[k] for k in ['RMSE','combined','maps','precision','warm_mean_seconds_both_maps']}),flush=True)
