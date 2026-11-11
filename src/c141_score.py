"""C141 TRAIN diagnostic scoring. No fitted parameters. Uniform point weights.
Separates exact area LOS, numerical receiver detection, and log/power export.
"""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/"audit/c141_direct_receiver"
panel=json.loads((A/"fit_panel.json").read_text())["records"]
C=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458.)
def pathloss(x):
 v=np.full_like(x,np.inf,dtype=float);m=x>0;v[m]=-10*np.log10(x[m]);return v
def metric(p,y,sel):
 w=np.broadcast_to(sel[:,None],p.shape);finite=np.isfinite(p)&w
 e=np.where(finite,np.where(np.isfinite(p),p,0)-y,0)
 d=finite.sum(0)
 return {"RMSE":np.sqrt((e**2).sum(0)/np.maximum(d,1)).tolist(),
 "MAE":(np.abs(e).sum(0)/np.maximum(d,1)).tolist(),"finite_points":d.tolist(),"nonfinite_points":(w&~np.isfinite(p)).sum(0).tolist()}
records=[];allpred={};allY=[];allB=[];allcase=[];allgroup=[];area_gap=[];native_gap=[]
for rec in panel:
 f=A/f"fit_replay/{rec['map']}.npz"
 if not f.exists():continue
 z=np.load(f);g=np.load(R/f"audit/c136_native/cache/{rec['map']}.npz")
 pos=np.array(rec["cache_positions"]);inv=g["inv"][pos];ids=g["uf"][inv];w=z["weights"]
 assert np.array_equal(ids,z["ids"])
 dw=g["dw"][inv].astype(float);D=z["area4_native"][...,None]
 assert np.all((dw>0)|(w[:,:,None]<1e-7))
 extra=np.divide(D*(1-dw),dw,out=np.zeros_like(dw),where=dw>0)
 extra=np.maximum(extra,0)
 old=D+extra
 loss0=pathloss((old*w[...,None]).sum(1))+C
 predictions={"physical_old_linear":loss0,"physical_old_log":(np.where(w[...,None]>1e-12,pathloss(old),0)*w[...,None]).sum(1)+C}
 for tag in ["native_full","native_first"]:
  total=z[tag+"_native"][...,None]+extra
  predictions[tag+"_linear"]=pathloss((total*w[...,None]).sum(1))+C
  # Zero receiver+zero background gives infinite PL; never conceal with a floor.
  cell=pathloss(total)
  cell=np.where(w[...,None]>1e-12,cell,0)
  predictions[tag+"_log"]=(cell*w[...,None]).sum(1)+C
 y=np.asarray(rec["labels"]);base=np.asarray(rec["C82"]);case=np.asarray(rec["is_case"])
 predictions["C82"]=base
 for tag in list(predictions):
  if tag!="C82":predictions["anchored_"+tag]=base+predictions[tag]-loss0
 for k,p in predictions.items():allpred.setdefault(k,[]).append(p)
 allY.append(y);allB.append(base);allcase.append(case);allgroup.extend([rec["group"]]*len(case))
 area_gap.append(pathloss(z["area256"])-pathloss(z["area4"]))
 native_gap.append(pathloss(z["native_full"])-pathloss(z["area256"]))
 record={"map":rec["map"],"sample_id":rec["sample_id"],"group":rec["group"],"points":len(case),"cases":int(case.sum()),
  "source_cache_direct_vs_independent_area_max_abs_dB":float(np.max(np.abs(np.asarray(rec["candidate_direct_PL"])-(pathloss(z["area4"])[:,None]+C)))),
  "df_quantization_check_max_abs_dB":float(np.max(np.abs(-g["df"][inv,7:9].astype(float)*100-(pathloss(D)+np.zeros_like(dw))))),
  "predictions":{k:p.tolist() for k,p in predictions.items()},
  "labels":y.tolist(),"is_case":case.tolist(),"pixels":rec["pixels"]}
 records.append(record)
if not records:raise SystemExit("No complete map yet")
Y=np.concatenate(allY);B=np.concatenate(allB);case=np.concatenate(allcase);group=np.array(allgroup)
P={k:np.concatenate(v) for k,v in allpred.items()}
summary={"maps":len(records),"total_maps":len(panel),"points":len(Y),"case_points":int(case.sum()),
 "scope":"FIT selected counterexamples plus ordinary controls, not an unbiased score, not held-out performance.",
 "training":False,"candidate_parameters_fitted":False,"groups":{}}
for name,sel in [("all",np.ones(len(Y),bool)),("case",case),("ordinary",~case),("C82_good_case",case&(group=="C82_good")),("shared_miss_case",case&(group=="shared_miss"))]:
 summary["groups"][name]={k:metric(v,Y,sel) for k,v in P.items()}
d=np.concatenate(area_gap);nd=np.concatenate(native_gap)
summary["area4_to_256_dB_abs_quantiles_50_90_99_max"]=np.quantile(np.abs(d),[.5,.9,.99,1]).tolist()
summary["native_full_vs_area256_zero_count"]=int((~np.isfinite(nd)).sum())
summary["native_full_vs_area256_finite_change_quantiles_50_90_99_max"]=np.quantile(nd[np.isfinite(nd)],[.5,.9,.99,1]).tolist()
summary["old_GPU_direct_vs_independent_CPU_area_max_abs_dB"]=max(x["source_cache_direct_vs_independent_area_max_abs_dB"] for x in records)
(A/"fit_scores.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
# Explicit nulls for undefined PL.
def clean(v):
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
(A/"fit_cases.json").write_text(json.dumps(clean(records),indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="groups"}))
for name,d in summary["groups"].items():
 print(name,{k:[round(x,4) for x in d[k]["RMSE"]] for k in ["C82","physical_old_linear","physical_old_log","native_full_linear","native_full_log","anchored_native_full_linear","anchored_native_full_log"]})
