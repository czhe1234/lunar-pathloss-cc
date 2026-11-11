"""Independent C28 full-pixel scoring, focal-vs-other tradeoff, and frozen gate."""
from pathlib import Path
import json,csv,time
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/"audit/c28_matched_exposure_20261101";D=R.parent/"data"
def main():
 p=json.loads((A/"protocol.json").read_text());ids=p["fit_panel"];anchors=set(p["anchors"])
 select=np.array([i in anchors for i in ids]);Y=[np.load(D/f"train_pl{b}.npy",mmap_mode="r") for b in ["415","58"]];M=[np.load(D/f"train_mask{b}.npy",mmap_mode="r") for b in ["415","58"]]
 runs={"parent":"c24_fit128_standard","anchor_only":"c28_anchor_only","joint":"c28_joint","project_other":"c28_project_other"}
 result={"scope":"128 FIT maps, focal4oversampled; not officialVAL or unseen terrain generalization","models":{},"checks":{}}
 completions={};all_sse=[];count=None
 for name,runname in runs.items():
  run=R/"runs"/runname;dest=run/"fit_complete";assert json.loads((dest/"ids.json").read_text())==ids
  if name!="parent":completions[name]=json.loads((run/"complete.json").read_text())
  pred=np.load(dest/"pred.npy",mmap_mode="r");ss=np.zeros((len(ids),2));cc=np.zeros_like(ss)
  for j,i in enumerate(ids):
   for b in [0,1]:
    mask=np.unpackbits(M[b][i],bitorder="big").reshape(256,256).astype(bool)
    e=pred[j,b].astype(np.float64)-Y[b][i].astype(np.float64)
    ss[j,b]=np.square(e[mask]).sum();cc[j,b]=mask.sum()
  if count is None:count=cc
  else:assert np.array_equal(count,cc)
  groups={}
  for k,sel in [("anchor",select),("other",~select),("all",np.ones(len(ids),bool))]:
   s=ss[sel].sum(0);c=cc[sel].sum(0);rmse=np.sqrt(s/c)
   groups[k]={"rmse":rmse.tolist(),"score":float(rmse.mean()),"sse":s.tolist(),"count":c.astype(int).tolist()}
  reported=json.loads((dest/"metrics.json").read_text())["rmse"]
  delta=float(np.max(np.abs(np.array(groups["all"]["rmse"])-reported)));assert delta<1e-8
  result["models"][name]=groups;result["checks"][name]={"independent_max_difference_db":delta};all_sse.append(ss)
 for key in ["initial_state_sha256","map_order_sha256","query_order_sha256"]:
  assert len({d[key] for d in completions.values()})==1
  result["checks"][key+"_equal"]=True
 result["gradient_diagnostics"]={n:{k:c[k] for k in ["negative_gradient_fraction","mean_gradient_cosine","clipped_steps"]} for n,c in completions.items()}
 diff=np.array(result["models"]["project_other"]["anchor"]["rmse"])-result["models"]["joint"]["anchor"]["rmse"]
 all_diff=result["models"]["project_other"]["all"]["score"]-result["models"]["joint"]["all"]["score"]
 result["projection_minus_joint_anchor_db"]=diff.tolist();result["projection_minus_joint_all_mean_db"]=all_diff
 result["advance_full_data_projection"]=bool(np.all(diff<=-.1) and all_diff<=.02)
 (A/"full_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
 np.savez(A/"paired_per_map.npz",ids=np.array(ids),sse=np.stack(all_sse),count=count,model_names=np.array(list(runs)))
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
