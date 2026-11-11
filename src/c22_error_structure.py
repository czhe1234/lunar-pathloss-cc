"""Label-based diagnostic only: decompose parent FIT errors by spatial scale."""
from pathlib import Path
import json, numpy as np
from scipy.ndimage import gaussian_filter
R=Path(__file__).resolve().parents[1]
D=R.parent/"data";A=R/"audit/c22_scale_diagnosis_20261101"
path=R/"runs/c21r1_pair_s20261123/fit_panel_A"
ids=json.loads((path/"ids.json").read_text())
pred=np.load(path/"pred.npy",mmap_mode="r")
result={"scope":"C21r1 A selected15000,128 FIT maps,full valid pixels; label-based decomposition is diagnostic, not a usable predictor","bands":{}}
for band_idx,band in enumerate(["415","58"]):
 y=np.load(D/f"train_pl{band}.npy",mmap_mode="r")
 masks=np.load(D/f"train_mask{band}.npy",mmap_mode="r")
 total_sse=0.;total_count=0;bias_sse=0.;corrected={s:0. for s in [1,2,4,8,16]}
 gradss=np.zeros(4,np.float64);gradcount=0
 for j,i in enumerate(ids):
  yy=np.asarray(y[i],np.float64);p=np.asarray(pred[j,band_idx],np.float64)
  m=np.unpackbits(masks[i],bitorder="big").reshape(256,256).astype(np.float64)
  e=(p-yy)*m;n=m.sum()
  total_sse+=np.square(e).sum();total_count+=n;bias_sse+=e.sum()**2/n
  for s in corrected:
   low=gaussian_filter(e,sigma=s,mode="nearest")/gaussian_filter(m,sigma=s,mode="nearest").clip(1e-6)
   corrected[s]+=np.square((e-low)*m).sum()
  for axis in [0,1]:
   gp=np.diff(p,axis=axis);gy=np.diff(yy,axis=axis)
   if axis==0:valid=m[:-1]*m[1:]
   else:valid=m[:,:-1]*m[:,1:]
   gradss+=np.array([(gp*gp*valid).sum(),(gy*gy*valid).sum(),(gp*gy*valid).sum(),((gp-gy)**2*valid).sum()])
   gradcount+=valid.sum()
 result["bands"][band]={"rmse":float(np.sqrt(total_sse/total_count)),
   "fraction_sse_removed_by_oracle_per_map_constant":float(bias_sse/total_sse),
   "rmse_after_oracle_per_map_constant":float(np.sqrt((total_sse-bias_sse)/total_count)),
   "rmse_after_oracle_smooth_error_removal":{str(s):float(np.sqrt(v/total_count)) for s,v in corrected.items()},
   "gradient_pred_rms":float(np.sqrt(gradss[0]/gradcount)),"gradient_target_rms":float(np.sqrt(gradss[1]/gradcount)),
   "gradient_uncentered_cosine":float(gradss[2]/np.sqrt(gradss[0]*gradss[1])),
   "gradient_error_rmse":float(np.sqrt(gradss[3]/gradcount)),"count":int(total_count)}
(A/"parent_error_structure.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
