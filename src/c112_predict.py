"""Frozen input-only C82 maps for metadata-selected near-TX controls."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np,csv
from src.c82_inference import Predictor
A=R/'audit/c112_TXresponse'
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--shard',type=int,default=0);a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(4);rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));h=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 plan=json.loads((A/'protocol.json').read_text());net=Predictor(A/'interface');t=time.monotonic()
 if a.check:
  ids=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());ref=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r');diff=[];which=[0,2,4,6]
  for pos in which:
   r=rows[ids[pos]];p=net.predict(h[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']));diff.append(p.astype(np.float64)-ref[pos])
  diff=np.array(diff);rec=dict(input_ids=[ids[j] for j in which],pixel_RMS=np.sqrt((diff**2).mean((0,2,3))).tolist(),max_abs=float(abs(diff).max()),GPU=gpu,seconds=time.monotonic()-t)
  atomic_json(A/'replay.json',rec);assert max(rec['pixel_RMS'])<.03,rec
  print(json.dumps(rec),flush=True);return
 assert (A/'replay.json').exists() and max(json.loads((A/'replay.json').read_text())['pixel_RMS'])<.03
 ids=plan['ids'][a.shard::2];out=A/f'shard{a.shard}';out.mkdir(exist_ok=False);atomic_json(out/'ids.json',ids)
 array=np.lib.format.open_memmap(out/'prediction.npy',mode='w+',dtype='float32',shape=(len(ids),2,256,256))
 for j,i in enumerate(ids):
  r=rows[i];array[j]=net.predict(h[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']))
  assert np.isfinite(array[j]).all()
  if (j+1)%10==0 or j+1==len(ids):
   rec=dict(maps=j+1,total=len(ids),seconds=time.monotonic()-t);atomic_json(out/'progress.json',rec);print(rec,flush=True)
 array.flush();atomic_json(out/'done.json',dict(maps=len(ids),GPU=gpu,seconds=time.monotonic()-t,input_only=True,labels_or_masks_opened=False))
if __name__=='__main__':main()
