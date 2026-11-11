"""Score the SAME saved predictions independently with torch; no model inference."""
import os,sys,fcntl,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,atomic_json
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);args=p.parse_args();out=ROOT/args.predictions
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=gpu_guard();torch.set_num_threads(8)
ids=json.loads((out/'indices.json').read_text());ii=np.array(ids['index_rows']);split=ids['source_split'];bands={};start=time.time()
with torch.inference_mode():
 for band in ('415','58'):
  pred=np.load(out/f'pl{band}.npy',mmap_mode='r');target=np.load(DATA/f'{split}_pl{band}.npy',mmap_mode='r');mask=np.load(DATA/f'{split}_mask{band}.npy',mmap_mode='r')
  total32=torch.zeros((),dtype=torch.float64,device='cuda');total64=total32.clone();count=total32.clone()
  for lo in range(0,len(ii),64):
   ids=ii[lo:lo+64];a=torch.from_numpy(np.array(pred[lo:lo+len(ids)],copy=True)).cuda();b=torch.from_numpy(np.array(target[ids],copy=True)).cuda().float();m=torch.from_numpy(np.unpackbits(mask[ids],axis=1,bitorder='big').reshape(len(ids),256,256)).cuda().bool()
   error32=(a-b).square();error64=(a.double()-b.double()).square()
   total32+=error32.masked_select(m).sum(dtype=torch.float64);total64+=error64.masked_select(m).sum(dtype=torch.float64);count+=m.sum(dtype=torch.float64)
  bands[band]={'rmse_fp32_error_fp64_sum':float(torch.sqrt(total32/count)),'rmse_fp64_error_fp64_sum':float(torch.sqrt(total64/count)),'valid_pixels':int(count),'sse_fp64':float(total64)}
summary={'predictions':str(out.relative_to(ROOT)),'no_model_forward':True,'bands':bands,'seconds':time.time()-start,'device':device};atomic_json(out/'same_prediction_torch_score.json',summary);print(json.dumps(summary),flush=True)
