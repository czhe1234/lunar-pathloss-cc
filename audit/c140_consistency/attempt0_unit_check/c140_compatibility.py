"""C140 CPU-only compatibility audit. No training and no official VAL access."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="1"
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit/c140_consistency"
DATA=ROOT.parent/"data"
def dump(name,value):
 p=OUT/name; t=p.with_suffix(p.suffix+".tmp")
 t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def pl(g):
 x=np.asarray(g,dtype=np.float64)
 z=np.full_like(x,np.nan);pos=x>0;z[pos]=-10*np.log10(x[pos]);return z
def safe_list(x):return [float(z) if np.isfinite(z) else None for z in np.atleast_1d(x)]
def metric(pred,y,w):
 valid=np.isfinite(pred)&np.isfinite(y)&(w>0)
 den=np.sum(w*valid,axis=0);sse=np.sum(np.where(valid,(np.nan_to_num(pred)-y)**2,0)*w,axis=0)
 return {"rmse":safe_list(np.sqrt(sse/np.maximum(den,1e-300))),"SSE":safe_list(sse),"valid_weight":safe_list(den),
         "excluded_weight":safe_list(np.sum(w*(~np.isfinite(pred)),axis=0))}
def quant(x,mask):
 return [safe_list(np.quantile(x[mask[:,b],b],[.5,.9,.99,1])) if mask[:,b].any() else [] for b in range(2)]
def describe(pred,y,w,base,vis):
 e=base-y; tail=(np.abs(e)>=5);vp=np.isfinite(pred)&(w>0)
 corr=pred-base; right=(corr*(-e)>0)
 ans={"all":metric(pred,y,w),"tail":metric(pred,y,w*tail),
      "clear":metric(pred,y,w*(vis[:,None]>=.999)),
      "not_clear":metric(pred,y,w*(vis[:,None]<.999))}
 for label,sel in [("tail",tail),("all",np.ones_like(tail))]:
  ww=w*vp*sel; d=ww.sum(0)
  ans[label+"_correction_direction_accuracy"]=safe_list((ww*right).sum(0)/np.maximum(d,1e-300))
  ans[label+"_direction_majority_baseline"]=safe_list(np.maximum((ww*(e>0)).sum(0),(ww*(e<0)).sum(0))/np.maximum(d,1e-300))
 return ans
def global_offset(pred,y,w):
 valid=np.isfinite(pred); ww=w*valid
 return (ww*np.where(valid,y-np.nan_to_num(pred),0)).sum(0)/ww.sum(0)
def floor_direct(d,y,w):
 # Any nonnegative additional paths imply PL_total <= PL_direct.
 bound=pl(d)
 err=np.where(d>0,np.maximum(y-np.nan_to_num(bound),0),0)
 return {"optimistic_RMSE_floor":safe_list(np.sqrt((w*err**2).sum(0)/w.sum(0))),
 "fraction_violating_by_over_1dB":safe_list((w*(err>1)).sum(0)/w.sum(0)),
 "fraction_violating_by_over_3dB":safe_list((w*(err>3)).sum(0)/w.sum(0)),
 "positive_violation_quantiles_50_90_99_max_dB":quant(err,(w>0)&(err>0))}
def local_order(pred,y,w,ids,pixels,pp):
 n=np.zeros((2,));good=np.zeros(2);nonflat=np.zeros(2);right=np.zeros(2)
 for i,mid in enumerate(ids):
  s=slice(i*pp,(i+1)*pp);pix=pixels[s]
  pairs=cKDTree(np.stack([pix//256,pix%256],1)).query_pairs(2.01,output_type="ndarray")
  if not len(pairs):continue
  a,b=pairs.T
  dy=y[s][a]-y[s][b]; dp=pred[s][a]-pred[s][b]
  ww=np.minimum(w[s][a],w[s][b])*np.isfinite(dp)
  d=np.nan_to_num(dp); valid=np.abs(dy)>=3
  n+=(ww*valid).sum(0);good+=(ww*valid*(d*dy>0)).sum(0)
  nonflat+=(ww*valid*(np.abs(d)>=1)).sum(0)
 return {"pair_distance_max_m":2.01,"label_contrast_min_dB":3,
 "weighted_pairs":safe_list(n),"correct_order_fraction":safe_list(good/np.maximum(n,1e-300)),
 "predicted_contrast_at_least_1dB_fraction":safe_list(nonflat/np.maximum(n,1e-300))}
rows=list(csv.DictReader((DATA/"train_index.csv").open()))
protocol=json.loads((ROOT/"audit/c136_native/protocol.json").read_text())
assert not(set(rows[i]["terrain_id"] for i in protocol["fit"])&set(rows[i]["terrain_id"] for i in protocol["monitor"]))
raw_y=[np.load(DATA/f"train_pl{b}.npy",mmap_mode="r") for b in ["415","58"]]
raw_m=[np.load(DATA/f"train_mask{b}.npy",mmap_mode="r") for b in ["415","58"]]
results={"scope":"C136 FIT 248 maps / original monitor 628 maps, C101 fixed pixels. TRAIN only.",
 "created":"2026-11-10","uses_official_VAL":False,"models_trained":False,"path_order":["D"]+[f"S{k}" for k in range(1,7)]+["diffuse","spec_before_diffuse","spec_after_diffuse","spec_before_and_after_diffuse"],
 "notes":["All raw physical metrics use absolute gain, not an anchor that cancels physical errors.",
 "Calibration is one FIT-only dB offset per band, not a scene-wise fit.",
 "Necessary positivity bound is specific to the approximate direct operator, not a law-level contradiction.",
 "Refinement differences are diagnostics, not confidence bounds or exact integration errors."]}
datasets={}; locations={}
for split in ["fit","monitor"]:
 start=time.time();ids=protocol[split];folder=ROOT/"audit/c101_residual"/split
 meta=json.loads((folder/"done.json").read_text());pp=meta["points_per_map"];index={int(mid):i for i,mid in enumerate(meta["ids"])}
 def arr(name,d):return np.load(folder/(name+".npy"),mmap_mode="r").reshape(len(index),pp,*d)
 yy=arr("target",(4,));bb=arr("base",(2,));ww=arr("weight",());px=arr("pixel",())
 out={k:[] for k in ["y","w","base","pixels","power","vis"]}
 for j,mid in enumerate(ids):
  i=index[mid]; y=np.array(yy[i]);pixels=np.array(px[i])
  z=np.load(ROOT/f"audit/c135_paths/cache/{mid}.npz")
  assert np.array_equal(z["pixels"],pixels)
  power=z["power"].astype(np.float64)
  assert np.isfinite(power).all() and (power>=0).all()
  for b in range(2):
   label=np.array(raw_y[b][mid]).reshape(-1)[pixels].astype(np.float32)
   mask=np.unpackbits(raw_m[b][mid],bitorder="big")[pixels]
   assert np.array_equal(mask,y[:,b+2]),("mask",split,mid,b)
   assert np.array_equal(label[mask>0],y[:,b][mask>0]),("label",split,mid,b)
  out["y"].append(y[:,:2]);out["w"].append(y[:,2:]*np.array(ww[i])[:,None])
  out["base"].append(np.array(bb[i]));out["pixels"].append(pixels)
  out["power"].append(power);out["vis"].append(z["direct_geometry"][:,0].astype(float))
  locations[mid]=(split,j)
 for k in out:out[k]=np.concatenate(out[k])
 out["ids"]=ids;out["pp"]=pp;datasets[split]=out
 results[split]={"maps":len(ids),"terrains":len(set(rows[i]["terrain_id"] for i in ids)),"points":len(out["y"]),
 "raw_cache_labels_and_masks_exact":True,"C82":metric(out["base"],out["y"],out["w"]),"load_seconds":time.time()-start}
 print(json.dumps({"event":"data_checked","split":split,"maps":len(ids),"seconds":time.time()-start}),flush=True)

# Absolute C135 physical metrics, FIT-only scale.
for name,ch in [("direct_only",[0]),("old_D_S1_diffuse",[0,1,7]),("all_11_paths",list(range(11)))]:
 gains={s:d["power"][:,ch].sum(1) for s,d in datasets.items()}
 predictions={s:pl(g) for s,g in gains.items()}
 offset=global_offset(predictions["fit"],datasets["fit"]["y"],datasets["fit"]["w"])
 results.setdefault("C135_absolute",{})[name]={"FIT_only_dB_offset":safe_list(offset)}
 for s,d in datasets.items():
  p=predictions[s]
  results["C135_absolute"][name][s]={
   "raw":describe(p,d["y"],d["w"],d["base"],d["vis"]),
   "FIT_scaled":describe(p+offset,d["y"],d["w"],d["base"],d["vis"]),
   "local_order":local_order(p,d["y"],d["w"],d["ids"],d["pixels"],d["pp"])}
for s,d in datasets.items():
 results[s]["candidate_direct_positivity_bound"]=floor_direct(d["power"][:,0],d["y"],d["w"])
dump("compatibility_partial.json",results)
print(json.dumps({"event":"absolute_metrics_done"}),flush=True)

# Paired quadrature refinement: no choice based on labels, all existing intersections.
refinements=[]
for left,right in [("cache","fine"),("fine","mis1024"),("mis1024","mis8192"),("mis1024","spec32"),("spec32","spec64")]:
 leftdir=ROOT/"audit/c135_paths"/left;rightdir=ROOT/"audit/c135_paths"/right
 ids=sorted({int(p.stem) for p in leftdir.glob("*.npz")}&{int(p.stem) for p in rightdir.glob("*.npz")}&set(locations))
 ww=[];yy=[];pa=[];pb=[];gb=[];zs=[];splits=[]
 for mid in ids:
  a=np.load(leftdir/f"{mid}.npz");b=np.load(rightdir/f"{mid}.npz")
  assert np.array_equal(a["pixels"],b["pixels"])
  split,j=locations[mid];d=datasets[split];sl=slice(j*d["pp"],(j+1)*d["pp"])
  ww.append(d["w"][sl]);yy.append(d["y"][sl]);pa.append(a["power"].astype(float));pb.append(b["power"].astype(float));splits.append(split)
 if not ids:continue
 w=np.concatenate(ww);y=np.concatenate(yy);a=np.concatenate(pa);b=np.concatenate(pb)
 item={"left":left,"right":right,"ids":ids,"scope_counts":{s:splits.count(s) for s in ["fit","monitor"]},"weight":safe_list(w.sum(0))}
 for name,ch in [("total",list(range(11))),("D",[0]),("all_specular",list(range(1,7))),("all_diffuse",list(range(7,11)))]:
  ga=a[:,ch].sum(1);gb=b[:,ch].sum(1);p1=pl(ga);p2=pl(gb);valid=(w>0)&np.isfinite(p1)&np.isfinite(p2)
  err=np.abs(p1-p2)
  item[name]={"paired_dB_difference":metric(p1,p2,np.where(np.isfinite(p2),w,0)),
   "absolute_dB_difference_quantiles_50_90_99_max":quant(err,valid),
   "left_nonpositive_weight":safe_list((w*(ga<=0)).sum(0)),"right_nonpositive_weight":safe_list((w*(gb<=0)).sum(0))}
 item["right_raw_RMSE"]=metric(pl(b.sum(1)),y,w)
 item["right_direct_bound"]=floor_direct(b[:,0],y,w)
 refinements.append(item)
results["refinements"]=refinements
dump("compatibility_partial.json",results)
print(json.dumps({"event":"refinements_done"}),flush=True)

# C132 coarse positive K: exact deterministic fixed-point replay, no learning.
graphs={}
def graph(terrain):
 if terrain not in graphs:
  g=np.load(ROOT/f"audit/c132_transport/cache/{terrain}/graph.npz")
  src=g["src"];dst=g["dst"];k=g["k"].astype(float)
  mats=[csr_matrix((k[:,b],(dst,src)),shape=(2048,2048)) for b in range(2)]
  colsum=np.stack([np.asarray(m.sum(0)).ravel() for m in mats],1)
  assert (k>=0).all() and colsum.max()<1
  graphs[terrain]=(mats,np.tile(g["area"].astype(float),2),colsum.max(0))
 return graphs[terrain]
phys={};fixed_stats=[]
for split,d in datasets.items():
 out={k:[] for k in ["zero","three","converged"]};start=time.time()
 for j,mid in enumerate(d["ids"]):
  terrain=rows[mid]["terrain_id"];mats,area,kmax=graph(terrain)
  z=np.load(ROOT/f"audit/c132_transport/cache/{terrain}/{mid}.npz")
  sl=slice(j*d["pp"],(j+1)*d["pp"]);assert np.array_equal(z["pixels"],d["pixels"][sl])
  Q=z["sq"].astype(float);L=Q.copy();norm=np.maximum(np.abs(Q).sum(0),1e-300);L3=None
  for it in range(1,257):
   update=np.stack([mats[b]@L[:,b] for b in range(2)],1)
   new=Q+update
   rel=np.abs(new-L).sum(0)/norm
   L=new
   if it==3:L3=L.copy()
   if it>=3 and rel.max()<=1e-8:break
  assert it<256,("nonconverged",mid,rel)
  residual=L-Q-np.stack([mats[b]@L[:,b] for b in range(2)],1)
  relative=np.abs(residual).sum(0)/norm
  ri=z["ri"];aw=z["aw"].astype(float);rq=z["rq"].astype(float);ds=z["ds"].astype(float)
  for name,extra in [("zero",np.zeros_like(Q)),("three",L3-Q),("converged",L-Q)]:
   # Clamp only roundoff cancellation in L-Q, not received power.
   assert extra.min()>=-1e-20
   extra=np.maximum(extra,0)
   multi=extra[ri]/area[ri,None]
   gain=ds[:,:2]+ds[:,2:]+((rq+multi)*aw[...,None]).sum(1)
   out[name].append(gain)
  fixed_stats.append({"map":mid,"split":split,"iterations":it,"relative_L1_residual":safe_list(relative),
                      "maximum_column_sum":safe_list(kmax)})
  if (j+1)%100==0:print(json.dumps({"event":"fixed_point","split":split,"maps":j+1,"seconds":time.time()-start}),flush=True)
 phys[split]={k:np.concatenate(v) for k,v in out.items()}
results["C132_fixed_point"]={"maps":len(fixed_stats),
 "maximum_iterations":max(r["iterations"] for r in fixed_stats),
 "maximum_relative_residual":np.max([r["relative_L1_residual"] for r in fixed_stats],0).tolist(),
 "maximum_column_sum":np.max([r["maximum_column_sum"] for r in fixed_stats],0).tolist()}
for name in ["zero","three","converged"]:
 ppred={s:pl(d[name]) for s,d in phys.items()}
 off=global_offset(ppred["fit"],datasets["fit"]["y"],datasets["fit"]["w"])
 results["C132_fixed_point"][name]={"FIT_only_dB_offset":safe_list(off)}
 for s,d in datasets.items():
  results["C132_fixed_point"][name][s]={"raw":describe(ppred[s],d["y"],d["w"],d["base"],d["vis"]),
    "FIT_scaled":describe(ppred[s]+off,d["y"],d["w"],d["base"],d["vis"])}
for s,d in datasets.items():
 results["C132_fixed_point"].setdefault("truncation_difference",{})[s]=metric(pl(phys[s]["three"]),pl(phys[s]["converged"]),d["w"])
dump("fixed_point_maps.json",fixed_stats)
dump("compatibility.json",results)
# Store only pilot H/TX physical outputs, no new label split.
for s,d in phys.items():np.savez_compressed(OUT/f"c132_fixed_point_{s}.npz",ids=datasets[s]["ids"],**d)
manifest={"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "protocol_sha256":hashlib.sha256((OUT/"PLAN.md").read_bytes()).hexdigest(),
          "raw_files":{p.name:{"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns} for p in [DATA/"train_index.csv",*[DATA/f"train_{x}{b}.npy" for x in ["pl","mask"] for b in ["415","58"]]]}}
dump("manifest.json",manifest)
print(json.dumps({"event":"complete","output":str(OUT/"compatibility.json")}),flush=True)
