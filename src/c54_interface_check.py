"""Verify fresh H/TX features and the fixed-band public composite interface."""
import sys,csv,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c54_inference import Predictor
import numpy as np,torch
gpu=gpu_guard();pred=Predictor(R/'runs/c54_combo_dev')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
fitids=np.load(R/'features/c38_dense/fit_map_indices.npy');pix=np.load(R/'features/c38_dense/fit_pixel.npy',mmap_mode='r');X=np.load(R/'features/c38_dense/fit_x.npy',mmap_mode='r')
records=[]
for pos in [0,100]:
 i=int(fitids[pos]);row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);r=int(row['tx_row']);c=int(row['tx_col'])
 pred.set_terrain(h,r,c);fresh,fs=pred.features(h,r,c)
 indices=np.array(pix[pos*1024:(pos+1)*1024],np.int64);old=np.array(X[pos*1024:(pos+1)*1024],np.float32)
 dif=np.abs(fresh[torch.as_tensor(indices,device='cuda')].cpu().numpy()-old)
 assert float(dif.max())<1e-4,float(dif.max())
 torch.cuda.synchronize();t=time.perf_counter();a=pred.predict(h,r,c);torch.cuda.synchronize();seconds=time.perf_counter()-t
 torch.cuda.synchronize();t=time.perf_counter();b=pred.predict(h,r,c);torch.cuda.synchronize();warm=time.perf_counter()-t
 assert np.isfinite(a).all() and np.max(abs(a-b))==0
 records.append({'map_index':i,'source':'FIT H/TX only','features294_max_abs_difference_vs_saved_CPU':float(dif.max()),
                 'repeat_prediction_max_abs_db':float(np.max(abs(a-b))),'seconds_cold_both_maps':seconds,'seconds_warm_both_maps':warm})
atomic_json(R/'audit/c54_band_experts_20261102/interface.json',{'gpu':gpu,'records':records,'input_only':True,'labels_or_validity_masks_read':False,
 'precision':'FrozenC38 BF16batch4096; new correction FP32/noTF32'})
print(json.dumps(records),flush=True)
