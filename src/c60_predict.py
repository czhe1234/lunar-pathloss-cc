"""Input-only raw H/TX production for C54 selected checkpoints; no GT/masks."""
import sys,time,argparse,json,hashlib,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c60_inference import Predictor
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scope',choices=['dev64','remaining','all'],default='dev64')
p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1);a=p.parse_args()
gpu=gpu_guard();rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
allids=np.load(R/'features/c6_points/dev_map_indices.npy').tolist()
small=json.loads((R/'audit/c43_architecture_20261102/protocol.json').read_text())['dev64_ids']
ids=small if a.scope=='dev64' else ([i for i in allids if i not in set(small)] if a.scope=='remaining' else allids)
ids=ids[a.shard::a.of];ck=R/'runs'/a.run
net=Predictor(ck)
folder=R/'runs'/a.run/'bf16_compiled'/f'{a.scope}_shard{a.shard}of{a.of}';folder.mkdir(parents=True,exist_ok=False)
atomic_json(folder/'indices.json',{'index_rows':ids,'sample_ids':[rows[i]['sample_id'] for i in ids]})
out=[np.lib.format.open_memmap(folder/f'pl{b}.npy',mode='w+',dtype=np.float32,shape=(len(ids),256,256)) for b in ['415','58']]
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');t=time.time();times=[]
for k,i in enumerate(ids):
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32)
 torch.cuda.synchronize();start=time.perf_counter();pred=net.predict(h,int(row['tx_row']),int(row['tx_col']));torch.cuda.synchronize();times.append(time.perf_counter()-start)
 for b in range(2):out[b][k]=pred[b]
 if k%20==0 or k+1==len(ids):
  progress={'maps':k+1,'total':len(ids),'seconds':time.time()-t,'last_inference_seconds_both_maps':times[-1]}
  atomic_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
for v in out:v.flush()
meta={'complete':True,'maps':len(ids),'scope':a.scope,'shard':a.shard,'of':a.of,'gpu':gpu,'seconds':time.time()-t,
 'input_only':True,'no_target_or_validity_mask_read':True,'correction_precision':'BF16_torchcompile_default_fullgraph; tested numerical QA in compiled_runtime.json','teacher_precision':'BF16_batch4096_as_originalC38',
 'warm_mean_seconds_both_maps':float(np.mean(times[1:])) if len(times)>1 else None,'manifest_sha256':hashlib.sha256((ck/'manifest.json').read_bytes()).hexdigest()}
atomic_json(folder/'inference.json',meta);print(json.dumps(meta),flush=True)
