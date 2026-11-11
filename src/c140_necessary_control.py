"""C140 necessary-condition paired control. No network training or tuning.
lambda=0, 1, infinity are fixed before reading the control results.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]="2";os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/"audit/c140_consistency";S=R/"audit/c101_residual/monitor"
meta=json.loads((S/"done.json").read_text());ids=meta["ids"];pp=meta["points_per_map"]
y4=np.load(S/"target.npy").astype(np.float64).reshape(len(ids),pp,4)
y=y4[...,:2];w=y4[...,2:];base=np.load(S/"base.npy").astype(np.float64).reshape(len(ids),pp,2)
pixels=np.load(S/"pixel.npy").reshape(len(ids),pp)
rows=list(csv.DictReader((R.parent/"data/train_index.csv").open()))
factor=(299792458./np.array([415e6,5.775e9])/(4*np.pi))**2
bound=[];vis=[]
for mid in ids:
 z=np.load(R/f"audit/c135_paths/cache/{mid}.npz");g=z["power"][:,0,:].astype(float)*factor
 b=np.full_like(g,np.inf);m=g>0;b[m]=-10*np.log10(g[m]);bound.append(b);vis.append(z["direct_geometry"][:,0])
bound=np.stack(bound);vis=np.stack(vis)
unscaled=np.minimum(base,bound);soft=(base+unscaled)/2
def metric(p,ww=w):
 den=ww.sum((0,1));sse=((p-y)**2*ww).sum((0,1))
 return {"RMSE":np.sqrt(sse/den).tolist(),"SSE":sse.tolist(),"valid_count":den.tolist()}
terrains=sorted(set(rows[i]["terrain_id"] for i in ids));ti={t:j for j,t in enumerate(terrains)}
den=np.zeros((len(terrains),2));sse={n:np.zeros_like(den) for n in ["C82","lambda1","hard"]}
for j,mid in enumerate(ids):
 t=ti[rows[mid]["terrain_id"]];den[t]+=w[j].sum(0)
 for n,p in [("C82",base),("lambda1",soft),("hard",unscaled)]:sse[n][t]+=((p[j]-y[j])**2*w[j]).sum(0)
rng=np.random.default_rng(140041);boot={n:[] for n in ["lambda1","hard"]}
for _ in range(1000):
 ix=rng.integers(len(terrains),size=len(terrains));d=den[ix].sum(0)
 b=np.sqrt(sse["C82"][ix].sum(0)/d)
 for n in boot:boot[n].append(np.sqrt(sse[n][ix].sum(0)/d)-b)
result={"scope":"Same original 128-terrain monitor, 628 maps, fixed 2048 pixels/map, TRAIN holdout only.",
 "type":"Necessary-condition output projection; NOT the requested trained-network ablation.",
 "reason":"For fixed nonnegative direct contribution D, every nonnegative path sum satisfies G>=D, equivalently PL<=PL_D.",
 "soft_solution":"argmin_p (p-C82)^2 + lambda*relu(p-PL_D)^2; lambda=1 is (C82+min(C82,PL_D))/2.",
 "parameters_fixed_before_results":[0,1,"infinity"],"no_labels_used_to_form_predictions":True,
 "C82":metric(base),"lambda1":metric(soft),"hard":metric(unscaled),
 "bootstrap_RMSE_increase_95CI":{n:np.quantile(np.array(v),[.025,.975],axis=0).T.tolist() for n,v in boot.items()},
 "groups":{}}
for name,group in [("clear",vis[...,None]>=.999),("not_clear",vis[...,None]<.999),
 ("C82_tail",np.abs(base-y)>=5),("C82_good",np.abs(base-y)<=1)]:
 ww=w*group
 result["groups"][name]={"C82":metric(base,ww),"lambda1":metric(soft,ww),"hard":metric(unscaled,ww)}
result["fraction_C82_good_made_error_over_3dB"]=(w*(np.abs(base-y)<=1)*(np.abs(unscaled-y)>3)).sum((0,1)).tolist()
result["fraction_C82_good_made_error_over_3dB"]=(np.array(result["fraction_C82_good_made_error_over_3dB"])/(w*(np.abs(base-y)<=1)).sum((0,1))).tolist()
err=np.maximum(y-bound,0)
result["violation_by_distance"]={}
for low,high in [(0,5),(5,20),(20,100),(100,1000)]:
 group=np.zeros((len(ids),pp,1),dtype=bool)
 for j,mid in enumerate(ids):
  rr=rows[mid];px=pixels[j];dist=np.hypot(px//256-int(rr["tx_row"]),px%256-int(rr["tx_col"]))
  group[j,:,0]=(dist>=low)&(dist<high)
 ww=w*group;d=ww.sum((0,1))
 result["violation_by_distance"][f"{low}_{high}"]={"valid_count":d.tolist(),"positive_bound_RMSE":np.sqrt((ww*err**2).sum((0,1))/d).tolist(),"SSE":(ww*err**2).sum((0,1)).tolist()}
examples=[]
for band in range(2):
 score=np.where(w[...,band]>0,err[...,band],-1).reshape(-1)
 top=np.argsort(score)[-20:][::-1]
 for pos in top:
  j,k=divmod(int(pos),pp);px=int(pixels[j,k]);mid=ids[j]
  examples.append({"band":band,"map":mid,"sample_id":rows[mid]["sample_id"],"row":px//256,"col":px%256,
    "label_dB":float(y[j,k,band]),"C82_dB":float(base[j,k,band]),"candidate_direct_dB":float(bound[j,k,band]),"incompatibility_dB":float(err[j,k,band])})
with (A/"direct_counterexamples.csv").open("w") as f:
 writer=csv.DictWriter(f,fieldnames=examples[0].keys());writer.writeheader();writer.writerows(examples)
(A/"necessary_condition_control.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(result),flush=True)
