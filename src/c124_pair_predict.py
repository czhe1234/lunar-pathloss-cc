"""No fitting: infer the previously frozen C121 24-map counterexamples."""
import sys,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c124_spectrum import Spectrum
import numpy as np,torch
from catboost import CatBoostRegressor
A=R/'audit/c124_spectrum'
def main():
 g=gpu_guard();torch.set_num_threads(3)
 inp=json.load(open(A/'pair_inputs.json'));old=np.load(R/'audit/c121_pairs/monitor/pair_features.npz')['old'][inp['pair_indices'],:2]
 n=len(old);hist=np.empty((n,2,256),np.float32);total=np.empty((n,2,2),np.float32)
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));hh=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');e=Spectrum()
 for mid in inp['maps']:
  k=np.flatnonzero(np.asarray(inp['mapid'])==mid);pixels=np.asarray(inp['pixels'],dtype='int32')[k].reshape(-1)
  if len(k)==0:continue
  r=rows[mid];x,t=e.features(hh[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']),pixels)
  hist[k]=x.half().float().cpu().numpy().reshape(-1,2,256);total[k]=t.cpu().numpy().reshape(-1,2,2)
 for mode in ['total','spectrum']:
  h=hist if mode=='spectrum' else np.repeat(np.maximum(total-np.log10(128)/4,-9.5).astype('float16').astype('float32'),128,axis=2)
  x=np.concatenate([old,total,h],axis=2).reshape(-1,427)
  prediction=[]
  for b in range(2):
   m=CatBoostRegressor();m.load_model(str(R/f'runs/c124_{mode}/band{b}.cbm'))
   prediction.append(m.predict(x,thread_count=4).astype('float32').reshape(n,2))
  np.save(A/f'pair_delta_{mode}.npy',np.stack(prediction,axis=-1))
 atomic_json(A/'pair_prediction_done.json',dict(GPU=g,pairs=n,maps=len(inp['maps']),official_val_used=False,labels_or_masks_read_for_features=False,fit=False))
if __name__=='__main__':main()
