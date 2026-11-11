"""Fixed-checkpoint FP32 numerical diagnostic, with TF32 disabled. No fitting."""
import sys,json,csv,time,hashlib,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,check_source_unchanged,atomic_json
import numpy as np,torch
out=ROOT/'audit/c4_remaining_error_20261031';folder=out/'official_val';plan=json.loads((out/'selection.json').read_text());package=ROOT/'deliveries/c4_full_refit_s20261031'
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=gpu_guard();check_source_unchanged();ck=ROOT/'runs/c4_full_refit_s20261031/best.pt';sha=hashlib.sha256(ck.read_bytes()).hexdigest();assert sha==plan['frozen_checkpoint_sha256']
sys.path.insert(0,str(package));from inference import Predictor
p=Predictor(package);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
assert not p.model.training and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
ids=plan['precision_val_indices'];rows=list(csv.DictReader((DATA/'val_index.csv').open()));hm=np.load(DATA/'val_hm.npy',mmap_mode='r');fp=np.lib.format.open_memmap(folder/'strict_fp32_predictions.npy',mode='w+',dtype=np.float32,shape=(len(ids),2,256,256));t0=time.time()
with torch.inference_mode():
 assert not torch.is_grad_enabled()
 for lo in range(0,len(ids),8):
  ii=ids[lo:lo+8];rs=[rows[i] for i in ii];h=torch.from_numpy(np.array(hm[[int(r['hm_row']) for r in rs]],copy=True)).cuda();r=torch.tensor([int(a['tx_row']) for a in rs],device='cuda');c=torch.tensor([int(a['tx_col']) for a in rs],device='cuda');x,fs=p.inputs(h,r,c);pred=p.model(x)+fs;fp[lo:lo+len(ii)]=pred.cpu().numpy()
fp.flush();torch.cuda.synchronize();assert hashlib.sha256(ck.read_bytes()).hexdigest()==sha
original=ROOT/'runs/c4_full_refit_s20261031/predictions/val_full_bf16';result={'maps':len(ids),'device':device,'seconds':time.time()-t0,'tf32_enabled':False,'autocast_enabled':False,'checkpoint_before_after_sha256':sha,'model_updated':False,'bands':{}}
replay=np.load(folder/'bf16_replay.npy',mmap_mode='r')
for k,b in enumerate(['415','58']):
 y=np.load(DATA/f'val_pl{b}.npy',mmap_mode='r')[ids].astype(float);m=np.unpackbits(np.load(DATA/f'val_mask{b}.npy',mmap_mode='r')[ids],axis=1,bitorder='big').reshape(len(ids),256,256).astype(bool);old=np.load(original/f'pl{b}.npy',mmap_mode='r')[ids].astype(float);new=fp[:,k].astype(float)
 result['bands'][b]={'valid_pixels':int(m.sum()),'saved_bf16_rmse':float(np.sqrt(np.mean((old[m]-y[m])**2))),'strict_fp32_rmse':float(np.sqrt(np.mean((new[m]-y[m])**2))),'prediction_difference_rms':float(np.sqrt(np.mean((new[m]-old[m])**2))),'prediction_difference_max_abs':float(np.max(np.abs(new[m]-old[m]))),'bf16_batch_replay_difference_rms':float(np.sqrt(np.mean((replay[:,k][m]-old[m])**2)))}
atomic_json(out/'precision_audit.json',result);print(json.dumps(result),flush=True)
