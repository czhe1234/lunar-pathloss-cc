"""C141 necessary bounds for fixed TRAIN counterexamples. No parameter fitting."""
from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[1];A=R/"audit/c141_direct_receiver"
C=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458.)
def pl(p):
 with np.errstate(divide="ignore"):return -10*np.log10(p)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
records=[];methods={};Y=[];case=[];groups=[];direct_first=[];weakest_ids=[]
for rec in json.loads((A/"fit_panel.json").read_text())["records"]:
 z=np.load(A/f"fit_replay/{rec['map']}.npz");w=z["weights"];pos=w>1e-12
 y=np.asarray(rec["labels"]);c=np.asarray(rec["is_case"],bool)
 cell=pl(z["native_full_native"])
 # All convex export rules, including power-domain and dB-domain interpolation,
 # remain no weaker than the weakest positive-weight donor direct component.
 weakest=np.max(np.where(pos,cell,-np.inf),axis=1)[:,None]+C
 pred={"area256_linear":pl(z["area256"])[:,None]+C,
       "native_full_linear":pl(z["native_full"])[:,None]+C,
       "native_full_log":np.sum(np.where(pos,cell,0)*w,axis=1)[:,None]+C,
       "native_full_weakest_donor":weakest,
       "native_first_linear":pl(z["native_first"])[:,None]+C}
 for k,v in pred.items():methods.setdefault(k,[]).append(v)
 Y.append(y);case.append(c);groups.extend([rec["group"]]*len(c))
 for j in range(len(c)):
  row={"map":rec["map"],"sample_id":rec["sample_id"],"pixel":int(rec["pixels"][j]),"row":int(rec["pixels"][j]//256),"col":int(rec["pixels"][j]%256),
       "group":rec["group"],"is_case":bool(c[j]),"label":y[j].tolist(),"C82":rec["C82"][j],
       "donor_ids":z["ids"][j].tolist(),"weights":w[j].tolist(),
       "native_full_donor_pathloss":(cell[j,:,None]+C).tolist(),
       "predictions":{k:v[j].tolist() for k,v in pred.items()},
       "weakest_donor_violation":(y[j]-weakest[j]).tolist()}
  records.append(row)
Y=np.concatenate(Y);case=np.concatenate(case);groups=np.asarray(groups);P={k:np.concatenate(v) for k,v in methods.items()}
out={"scope":"16 FIT terrains; enriched counterexamples and ordinary controls, not validation RMSE",
     "threshold_dB":1.0,"n_points":len(Y),"n_case":int(case.sum()),"groups":{}}
for group,sel in [("all",np.ones(len(Y),bool)),("case",case),("ordinary",~case),
                 ("C82_good_case",case&(groups=="C82_good")),("shared_miss_case",case&(groups=="shared_miss"))]:
 out["groups"][group]={"points":int(sel.sum()),"methods":{}}
 for k,p in P.items():
  finite=np.isfinite(p)&sel[:,None];violation=(Y-p)>1.
  delta=Y-p
  out["groups"][group]["methods"][k]={"finite":finite.sum(0).tolist(),
    "zero_power_uninformative":(sel[:,None]&~np.isfinite(p)).sum(0).tolist(),
    "violations_above_1dB":(violation&sel[:,None]).sum(0).tolist(),
    "worst_finite_violation_dB":[float(np.max(delta[finite[:,b],b])) if np.any(finite[:,b]) else None for b in range(2)]}
contradictions=[r for r in records if r["is_case"] and any(v>1 for v in r["weakest_donor_violation"])]
out["irreducible_by_fixed_donor_convex_interpolation_cases"]=len(contradictions)
out["decision"]="FIT compatibility fails; no monitor evaluation and no neural constraint training"
(A/"bounds.json").write_text(json.dumps(clean(out),indent=2,allow_nan=False)+"\n")
(A/"bound_cases.json").write_text(json.dumps(clean(records),indent=2,allow_nan=False)+"\n")
with (A/"remaining_finite_counterexamples.csv").open("w",newline="") as f:
 fields=["map","sample_id","row","col","group","label_415","label_58","weakest_direct_PL415","weakest_direct_PL58","violation_415","violation_58"]
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for r in contradictions:
  p=r["predictions"]["native_full_weakest_donor"];v=r["weakest_donor_violation"]
  writer.writerow(dict(map=r["map"],sample_id=r["sample_id"],row=r["row"],col=r["col"],group=r["group"],
   label_415=r["label"][0],label_58=r["label"][1],weakest_direct_PL415=p[0],weakest_direct_PL58=p[1],violation_415=v[0],violation_58=v[1]))
print(json.dumps(out,indent=2))
