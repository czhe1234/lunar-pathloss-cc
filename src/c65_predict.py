"""Frozen C65 H/TX-only predictions of complete fixed VAL; no target/mask reads."""
import sys,time,argparse,csv,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
from src.c60_inference import Predictor
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--of',type=int,default=3);a=p.parse_args()
gpu=gpu_guard();check_source_unchanged();run=R/'runs/c65_combo_full';net=Predictor(run)
rows=list(csv.DictReader((R.parent/'data/val_index.csv').open()));cuts=[0,900,1800,2330];ids=(list(range(cuts[a.shard],cuts[a.shard+1])) if a.of==3 else list(range(len(rows)))[a.shard::a.of]);assert len(rows)==2330
folder=run/'predictions'/f'val_full_c34_fresh_shard{a.shard}of{a.of}';folder.mkdir(parents=True,exist_ok=False)
atomic_json(folder/'indices.json',{'index_rows':ids,'sample_ids':[rows[i]['sample_id'] for i in ids],'source_split':'val'})
arrays=[np.lib.format.open_memmap(folder/f'pl{b}.npy',mode='w+',dtype=np.float32,shape=(len(ids),256,256)) for b in ['415','58']]
H=np.load(R.parent/'data/val_hm.npy',mmap_mode='r');times=[];t0=time.time()
for j,i in enumerate(ids):
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);torch.cuda.synchronize();t=time.perf_counter();pr=net.predict(h,int(row['tx_row']),int(row['tx_col']));torch.cuda.synchronize();times.append(time.perf_counter()-t)
 for b in range(2):arrays[b][j]=pr[b]
 if j%40==0 or j+1==len(ids):
  progress={'maps':j+1,'total':len(ids),'seconds':time.time()-t0,'last_dualband_seconds':times[-1]};atomic_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
for ar in arrays:ar.flush()
atomic_json(folder/'inference.json',{'complete':True,'maps':len(ids),'gpu':gpu,'input_only':True,'no_target_or_validity_mask_read':True,'precision':'BF16compiled; fixed frequency experts;teacher4096BF16','weights_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),'seconds':time.time()-t0,'warm_dualband_mean_seconds':float(np.mean(times[1:]))})
