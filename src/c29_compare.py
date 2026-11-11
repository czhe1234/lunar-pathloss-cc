"""C29 independent full-pixel verification and learning-rate control gate."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/"audit/c29_fit_regime_20261101";D=R.parent/"data"
def main():
 p=json.loads((A/"protocol.json").read_text());ids=p["fit128_base_protocol"]["fit_panel"]
 Y=[np.load(D/f"train_pl{b}.npy",mmap_mode="r") for b in ["415","58"]];M=[np.load(D/f"train_mask{b}.npy",mmap_mode="r") for b in ["415","58"]]
 runs={"low":"c24_fit128_standard","mid":"c29_fit128_mid_bf16","high":"c29_fit128_high_bf16"}
 result={"scope":"128 FIT full valid pixels, matched3000updates, only LR differs; not officialVAL","models":{},"checks":{}}
 hashes={};sse_all=[];count=None
 for name,runname in runs.items():
  run=R/"runs"/runname;dest=run/"fit_complete";c=json.loads((run/"complete.json").read_text());cfg=json.loads((run/"config.json").read_text())
  hashes[name]=[cfg["initial_state_sha256"],c["map_order_sha256"],c["query_order_sha256"]]
  assert json.loads((dest/"ids.json").read_text())==ids
  pred=np.load(dest/"pred.npy",mmap_mode="r");ss=np.zeros((len(ids),2));cc=np.zeros_like(ss)
  for j,i in enumerate(ids):
   for b in [0,1]:
    mask=np.unpackbits(M[b][i],bitorder="big").reshape(256,256).astype(bool);e=pred[j,b].astype(np.float64)-Y[b][i].astype(np.float64)
    ss[j,b]=np.square(e[mask]).sum();cc[j,b]=mask.sum()
  if count is None:count=cc
  else:assert np.array_equal(count,cc)
  rms=np.sqrt(ss.sum(0)/cc.sum(0));reported=json.loads((dest/"metrics.json").read_text())["rmse"]
  delta=float(np.max(np.abs(rms-reported)));assert delta<1e-8
  result["models"][name]={"rmse":rms.tolist(),"score":float(rms.mean()),"lr":cfg["protocol"]["lr"],"dev_panel":json.loads((run/"latest_eval.json").read_text())["DEV"]["rmse"]}
  result["checks"][name]={"independent_max_difference_db":delta};sse_all.append(ss)
 assert hashes["low"]==hashes["mid"]==hashes["high"]
 result["checks"]["initial_model_map_and_query_order_equal"]=True
 base=np.array(result["models"]["low"]["rmse"])
 result["delta_vs_low"]={k:(np.array(v["rmse"])-base).tolist() for k,v in result["models"].items() if k!="low"}
 result["advance_full_data_higher_lr"]=[k for k,v in result["delta_vs_low"].items() if max(v)<=-.1]
 result["hard_small"]=json.loads((R/"runs/c29_hard_small_original_regime/complete.json").read_text())
 old=json.loads((R/"runs/c21r1_small_a_s20261119/config.json").read_text());new=json.loads((R/"runs/c29_hard_small_original_regime/config.json").read_text())
 result["checks"]["hard_small_initial_model_matches_original"]=old["initial_state_sha256"]==new["initial_state_sha256"];assert result["checks"]["hard_small_initial_model_matches_original"]
 for k in ["seed","lr","steps","queries","chunk","eval_chunk","eval_every","geometry","resume"]:assert old[k]==new[k],k
 result["checks"]["hard_small_config_matches_original_except_FIT_ids"]=True
 (A/"full_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
 np.savez(A/"paired_per_map.npz",ids=np.array(ids),sse=np.stack(sse_all),count=count)
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
