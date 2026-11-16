"""Independent CPU C152 selected-prediction verification; no inference/training."""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.c150_report import load,sha,atomic,clean,statistics,parent_evidence,PARENT,paired_delta,evidence,align
from src.c151_report import run_context
AUDIT=ROOT/"audit/c152_fair_global"
RUN=ROOT/"runs/c152_c82_diffraction_global"
REGIONS=("NLoS","C145_NLoS_underprediction_ge5dB","c73_total_floor_coded")

def diagnostic(plan,context):
 ids=plan["panels"]["monitor"];data=ROOT.parent/"data"
 ys=[np.load(data/f"train_pl{b}.npy",mmap_mode="r") for b in ("415","58")]
 ws=[np.load(data/f"train_mask{b}.npy",mmap_mode="r") for b in ("415","58")]
 pred=np.load(RUN/"selected_monitor_prediction.npy",mmap_mode="r")
 ref=np.load(PARENT/"selected_monitor_prediction.npy",mmap_mode="r")
 base=np.load(ROOT/"audit/c88_mesh/base.npy",mmap_mode="r")
 bid=load(ROOT/"audit/c88_mesh/base_index.json");bid=bid["ids"] if isinstance(bid,dict) else bid
 bl={int(i):j for j,i in enumerate(bid)}
 profiles=np.load(ROOT/"audit/c149_error_audit/profiles.npy",mmap_mode="r")
 lookup={}
 for folder in ("c73_compact","c73_dev_extra_compact"):
  for p in sorted((ROOT/"features"/folder).glob("part*")):
   if not (p/"complete.json").exists():continue
   ii=load(p/"indices.json");v=np.load(p/"features.npy",mmap_mode="r")
   for j,i in enumerate(ii):
    if int(i) in ids:
     assert int(i) not in lookup;lookup[int(i)]=(v,j)
 n=len(ids);count=np.zeros((3,n,2),np.int64)
 sums={m:{k:np.zeros((3,n,2)) for k in ("SSE","error_sum")} for m in ("candidate","C145","C82")}
 for j,i in enumerate(ids):
  y=np.stack([a[i] for a in ys]).astype(np.float64)
  w=np.stack([np.unpackbits(a[i]).reshape(256,256) for a in ws]).astype(bool)
  a=np.asarray(ref[j],np.float64);highest=np.asarray(profiles[j,0])
  v,k=lookup[i];floor=np.asarray(v[k,[41,42]],np.float32)<=-3
  nlos=np.broadcast_to(highest>=0,w.shape)
  masks=(w&nlos,w&nlos&(a-y<=-5),w&floor)
  values={"candidate":np.asarray(pred[j],np.float64),"C145":a,"C82":np.asarray(base[bl[i]],np.float64)}
  for r,mask in enumerate(masks):
   count[r,j]=mask.sum((1,2))
   for m,value in values.items():
    e=value-y;sums[m]["SSE"][r,j]=np.where(mask,e*e,0).sum((1,2));sums[m]["error_sum"][r,j]=np.where(mask,e,0).sum((1,2))
 output={}
 for r,name in enumerate(REGIONS):
  c=count[r].sum(0);v={"count":c.tolist()}
  for m in sums:
   se=sums[m]["SSE"][r].sum(0);bias=sums[m]["error_sum"][r].sum(0)/c
   v[m]={"RMSE_dB":np.sqrt(se/c).tolist(),"MeanBias_dB":bias.tolist(),"SSE_dB2":se.tolist()}
  v["candidate_minus_C145_RMSE_dB"]=(np.asarray(v["candidate"]["RMSE_dB"])-v["C145"]["RMSE_dB"]).tolist()
  output[name]=v
 selected=context["selected"]["evaluation"]["monitor"]["diagnostic_regions"]
 for name,v in output.items():
  assert v["count"]==selected[name]["count"]
  for m in ("candidate","C145","C82"):
   assert np.allclose(v[m]["RMSE_dB"],selected[name][m]["RMSE"],rtol=1e-12,atol=1e-10)
   assert np.allclose(v[m]["MeanBias_dB"],selected[name][m]["MeanBias"],rtol=1e-12,atol=1e-10)
 return output

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--no-swanlab",action="store_true");args=parser.parse_args()
 plan=load(AUDIT/"protocol.json");ph=sha(AUDIT/"protocol.json")
 for f,h in plan["source_sha256"].items():assert sha(ROOT/f)==h,f
 assert plan["campaign"]=="C152" and plan["epochs"]==12 and plan["updates_per_arm"]==4224
 context=run_context(RUN,plan,ph,"candidate","C152")
 verified=evidence(context,hash_predictions=True)
 if verified is None:
  env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
  subprocess.run([str(ROOT/".venv/bin/python"),"-u","-m","src.c145_verify","--run",str(RUN.relative_to(ROOT)),"--tag","selected"],cwd=ROOT,env=env,check=True)
  verified=evidence(context,hash_predictions=True)
 assert verified is not None
 pv=parent_evidence();parent=statistics(PARENT/"selected_terrain_sse.npz",plan["panels"],pv)
 current=statistics(RUN/"selected_terrain_sse.npz",plan["panels"],verified);align(current,parent)
 st=current["monitor"];old=parent["monitor"]
 comparisons={"C145_selected12":paired_delta(st["terrain_sse"],old["terrain_sse"],st["terrain_count"]),
              "C82":paired_delta(st["terrain_sse"],old["terrain_C82_sse"],st["terrain_count"])}
 diag=diagnostic(plan,context)
 metric={p:{g:verified["panels"][p]["groups"][g] for g in verified["panels"][p]["groups"]} for p in ("fit","monitor")}
 result={"status":"verified","campaign":"C152","protocol_sha256":ph,"selected_epoch":context["selected"]["epoch"],"updates":4224,
  "selected":metric,"last_from_training_evaluation":context["history"][-1]["evaluation"],"paired_comparisons":comparisons,
  "gate_passed":all(v["both_band_upper_CI_strictly_below_zero"] for v in comparisons.values()),"automatic_budget_extension":False,
  "diagnostic_regions":diag,"diagnostic_original_training_values_independently_matched":True,
  "evidence":{"selected_verified_sha256":sha(RUN/"selected_verified.json"),"selected_terrain_sse_sha256":sha(RUN/"selected_terrain_sse.npz"),
   "predictor_sha256":context["export"]["model_sha256"],"checkpoint_sha256":context["export"]["checkpoint_sha256"],"report_source_sha256":sha(Path(__file__))},
  "limits":[plan["comparison_limit"],"Conditional terrain bootstrap on repeatedly used development monitor; not selection-corrected or blind.","Pointwise correlations/bias shifts do not establish physical noise or geometry causation."],
  "official_VAL_used":False,"training_or_network_inference_run_by_report":False}
 marker=AUDIT/"comparison_tracking.json"
 if not args.no_swanlab and not marker.exists():
  from src import cc_tracking
  run=cc_tracking.init(name="C152 verified comparison",campaign="C152",config={"run_kind":"analysis.paired_comparison","_cc_dashboard":{"kind":"analysis"},"protocol_sha256":ph},
   description="Single C82-preserving candidate vs frozen C145/C82, CPU saved-prediction evidence.",tags=["CPU","paired-terrain"])
  values={"gate/passed":int(result["gate_passed"]),"selection/epoch":result["selected_epoch"]}
  for ref,v in comparisons.items():
   for j,b in enumerate(("415MHz","5p775GHz")):
    values[f"{ref}/{b}/delta_RMSE_dB"]=v["point_delta_RMSE_dB"][j]
    values[f"{ref}/{b}/CI_low"]=v["delta_RMSE_dB_95CI"][j][0];values[f"{ref}/{b}/CI_high"]=v["delta_RMSE_dB_95CI"][j][1]
  cc_tracking.log(values,step=1);track={"run_id":run.id,"dir":str(run.dir)};cc_tracking.finish();atomic(marker,track)
 if marker.exists():result["tracking"]=load(marker)
 atomic(AUDIT/"comparison_summary.json",clean(result))
 lines=["# C152公平对照：已独立CPU核验","",f"选中Epoch {result['selected_epoch']}；仅单一候选，未重跑C145。","","|参照|candidate-reference 415/5.775 dB|415 CI|5.775 CI|双频通过|","|---|---|---|---|---|"]
 for ref,v in comparisons.items():lines.append(f"|{ref}|{v['point_delta_RMSE_dB']}|{v['delta_RMSE_dB_95CI'][0]}|{v['delta_RMSE_dB_95CI'][1]}|{v['both_band_upper_CI_strictly_below_zero']}|")
 lines+=["","|区域|C145 Bias 415/5.775|候选Bias|C145 RMSE|候选RMSE|","|---|---|---|---|---|"]
 for region,v in diag.items():lines.append(f"|{region}|{v['C145']['MeanBias_dB']}|{v['candidate']['MeanBias_dB']}|{v['C145']['RMSE_dB']}|{v['candidate']['RMSE_dB']}|")
 lines+=["",*result["limits"]]
 (AUDIT/"comparison_summary.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"status":"verified","selected_epoch":result["selected_epoch"],"gate_passed":result["gate_passed"],"paired_comparisons":comparisons}),flush=True)
if __name__=="__main__":main()
