"""Compare old/new errors in fixed input-only geometric zones on official VAL."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
new=R/'runs/c10_full_refit_s20261101/predictions/val_full_gpu_c10_fast_fresh';old=R/'runs/c4_full_refit_s20261031/predictions/val_full_bf16'
mapping=json.loads((new/'indices.json').read_text());om=json.loads((old/'indices.json').read_text());assert mapping['index_rows']==om['index_rows'];ii=np.array(mapping['index_rows']);zones=np.load(new/'geometry_zones.npy',mmap_mode='r')
names=['height_sensitive','clear_near_boundary','clear_grazing','clear_robust','shallow_blocked','deep_blocked'];result={}
for b in ['415','58']:
 p=np.load(new/f'pl{b}.npy',mmap_mode='r');q=np.load(old/f'pl{b}.npy',mmap_mode='r');y=np.load(D/f'val_pl{b}.npy',mmap_mode='r');mask=np.load(D/f'val_mask{b}.npy',mmap_mode='r')
 ss=np.zeros((2,6));cc=np.zeros(6,np.int64)
 for lo in range(0,len(ii),8):
  ids=ii[lo:lo+8];z=zones[lo:lo+len(ids)];m=np.unpackbits(mask[ids],axis=1,bitorder='big').reshape(len(ids),256,256).astype(bool);yy=y[ids].astype(float)
  ee=[(q[lo:lo+len(ids)].astype(float)-yy)**2,(p[lo:lo+len(ids)].astype(float)-yy)**2]
  for j in range(6):
   keep=m&(z==j);cc[j]+=keep.sum()
   for k in range(2):ss[k,j]+=ee[k][keep].sum()
 score=json.loads((new/'independent_score.json').read_text())['bands'][b];assert cc.sum()==score['valid_pixels'];assert abs((ss[1].sum()/cc.sum())**.5-score['rmse_db'])<1e-8
 result[b]={name:{'valid_pixels':int(cc[j]),'pixel_fraction':float(cc[j]/cc.sum()),'C4_rmse':float((ss[0,j]/cc[j])**.5),'C10_rmse':float((ss[1,j]/cc[j])**.5),'C10_SSE_fraction':float(ss[1,j]/ss[1].sum())} for j,name in enumerate(names)}
out={'scope':'All official fixed VAL pixels with per-band official masks; same nominal-geometry zone definitions for C4 and C10','zones_are_input_geometry_not_true_ray_states':True,'bands':result}
(R/'audit/c10_official_error_zones.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

