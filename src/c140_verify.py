"""Independent raw-TRAIN verification of C140's necessary-condition control."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]="2";os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/"audit/c140_consistency";S=R/"audit/c101_residual/monitor"
meta=json.loads((S/"done.json").read_text());ids=meta["ids"];pp=meta["points_per_map"]
base=np.load(S/"base.npy",mmap_mode="r").reshape(len(ids),pp,2);pix=np.load(S/"pixel.npy",mmap_mode="r").reshape(len(ids),pp)
yy=[np.load(R.parent/f"data/train_pl{b}.npy",mmap_mode="r") for b in ["415","58"]]
mm=[np.load(R.parent/f"data/train_mask{b}.npy",mmap_mode="r") for b in ["415","58"]]
factor=(299792458./np.array([415e6,5.775e9])/(4*np.pi))**2
SSE=np.zeros((4,2));count=np.zeros(2);groupSSE=np.zeros((3,2));groupN=np.zeros((3,2))
maskall=[]
for j,mid in enumerate(ids):
 z=np.load(R/f"audit/c135_paths/cache/{mid}.npz");g=z["power"][:,0,:].astype(float)*factor
 bound=np.full_like(g,np.inf);pos=g>0;bound[pos]=-10*np.log10(g[pos])
 b=base[j].astype(float);p0=b;p2=np.minimum(b,bound);p1=(p0+p2)/2
 y=np.stack([x[mid].reshape(-1)[pix[j]].astype(float) for x in yy],1)
 w=np.stack([np.unpackbits(x[mid],bitorder="big")[pix[j]] for x in mm],1).astype(bool)
 violation=np.maximum(y-bound,0)
 for k,p in enumerate([p0,p1,p2]):SSE[k]+=np.where(w,(p-y)**2,0).sum(0)
 SSE[3]+=np.where(w,violation**2,0).sum(0);count+=w.sum(0);maskall.append(w)
 v=z["direct_geometry"][:,0]
 for k,gr in enumerate([v>=.999,(v>0)&(v<.999),v<=0]):
  a=w&gr[:,None];groupN[k]+=a.sum(0);groupSSE[k]+=np.where(a,violation**2,0).sum(0)
ref=json.loads((A/"necessary_condition_control.json").read_text())
rmse=np.sqrt(SSE/count)
expected=np.array([ref[n]["RMSE"] for n in ["C82","lambda1","hard"]])
assert np.max(np.abs(rmse[:3]-expected))<1e-10
r=json.loads((A/"compatibility.json").read_text())["monitor"]["candidate_direct_positivity_bound"]["optimistic_RMSE_floor"]
assert np.max(np.abs(rmse[3]-r))<1e-10
con=np.load(A/"c132_receiver_converged_monitor.npz")
delta=np.full_like(con["PL"],np.nan);both=np.isfinite(con["PL3"])&np.isfinite(con["PL"])
np.subtract(con["PL3"],con["PL"],out=delta,where=both);valid=np.concatenate(maskall)&both
num={}
for band in range(2):
 d=np.abs(delta[valid[:,band],band]);p3=con["PL3"][valid[:,band],band]
 num[str(band)]={"paired_valid_count":int(len(d)),"absolute_dB_difference_quantiles_50_90_99_max":np.quantile(d,[.5,.9,.99,1]).tolist(),
 "fraction_difference_over_0_1_1_10_dB":[float(np.mean(d>x)) for x in [.1,1,10]],
 "fraction_large_changes_with_initial_physical_PL_over_250dB":float(np.mean(p3[d>1]>250)) if (d>1).any() else None}
result={"status":"passed","verification":"Independent streaming raw TRAIN labels/masks; no C101 target values used for scoring.",
 "reproduced_RMSE_C82_soft_hard_floor":rmse.tolist(),"valid_pixels":count.tolist(),
 "max_RMSE_disagreement":float(np.max(np.abs(rmse[:3]-expected))),
 "direct_bound_groups":{"clear":{"floor":np.sqrt(groupSSE[0]/groupN[0]).tolist(),"count":groupN[0].tolist()},
 "partial":{"floor":np.sqrt(groupSSE[1]/groupN[1]).tolist(),"count":groupN[1].tolist()},
 "blocked":{"floor":np.sqrt(groupSSE[2]/groupN[2]).tolist(),"count":groupN[2].tolist()}},
 "receiver_convergence_distribution":num}
(A/"verification.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(result),flush=True)
