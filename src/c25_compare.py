"""Independent FP64 scoring of C25 and terrain-group paired uncertainty."""
from pathlib import Path
import csv,json,time,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/"data";A=R/"audit/c25_map_batch_20261101"
def main():
 t0=time.time();old=np.load(R/"audit/c22_scale_diagnosis_20261101/paired_per_map.npz")
 ids=old["ids"];count=old["count"];rows=list(csv.DictReader((D/"train_index.csv").open()))
 Y=[np.load(D/f"train_pl{b}.npy",mmap_mode="r") for b in ["415","58"]]
 M=[np.load(D/f"train_mask{b}.npy",mmap_mode="r") for b in ["415","58"]]
 names=["C10","C22_batch8_low","C25_batch32_scaled","C25_batch64_scaled","C25_batch64_unscaled"]
 runnames=["c25_batch32_scaled","c25_batch64_scaled","c25_batch64_unscaled"]
 sse=[old["sse"][0],old["sse"][2]];checks={}
 ref=R/"runs/c22_full_low";refcomplete=json.loads((ref/"complete.json").read_text());refinit=json.loads((ref/"initialization.json").read_text())
 reforder=np.load(ref/"map_order.npy")
 for name,runname in zip(names[2:],runnames):
  run=R/"runs"/runname;dest=run/"dev_complete"
  assert json.loads((dest/"ids.json").read_text())==ids.tolist()
  complete=json.loads((run/"complete.json").read_text());init=json.loads((run/"initialization.json").read_text())
  assert complete["query_order_sha256"]==refcomplete["query_order_sha256"]
  assert np.array_equal(np.load(run/"map_order.npy"),reforder)
  assert init["model_sha256"]==refinit["model_sha256"]
  pred=np.load(dest/"pred.npy",mmap_mode="r");ss=np.zeros((len(ids),2))
  for j,i in enumerate(ids):
   for b in [0,1]:
    m=np.unpackbits(M[b][i],bitorder="big").reshape(256,256).astype(bool)
    assert int(m.sum())==count[j,b]
    e=pred[j,b].astype(np.float64)-Y[b][i].astype(np.float64);ss[j,b]=np.square(e[m]).sum()
  rmse=np.sqrt(ss.sum(0)/count.sum(0));reported=json.loads((dest/"metrics.json").read_text())
  diff=float(np.max(np.abs(rmse-np.array(reported["rmse"]))));assert diff<1e-8,(name,diff)
  checks[name]={"independent_max_difference_db":diff,"map_order_equal":True,"query_order_equal":True,"initial_model_equal":True,"selected_step":json.loads((run/"best.json").read_text())["step"]}
  sse.append(ss)
 sse=np.stack(sse);terrains=sorted(set(rows[int(i)]["terrain_id"] for i in ids));lookup={t:k for k,t in enumerate(terrains)}
 ts=np.zeros((len(names),len(terrains),2));tc=np.zeros((len(terrains),2))
 for j,i in enumerate(ids):
  k=lookup[rows[int(i)]["terrain_id"]];ts[:,k]+=sse[:,j];tc[k]+=count[j]
 draws=np.random.default_rng(20261202).integers(0,len(terrains),(2000,len(terrains)))
 bc=tc[draws].sum(1);boot=np.stack([np.sqrt(s[draws].sum(1)/bc) for s in ts]);rmse=np.sqrt(sse.sum(1)/count.sum(0))
 result={"scope":"CompleteDEV2562,612terrain groups; official formula, not officialVAL","count":count.sum(0).astype(int).tolist(),"models":{},"comparisons":{},"independent_checks":checks,"bootstrap":"2000 paired terrain draws, conditional on seed; not seed uncertainty"}
 for k,n in enumerate(names):result["models"][n]={"rmse":rmse[k].tolist(),"score":float(rmse[k].mean())}
 for ai,bi in [(2,0),(3,0),(4,0),(2,1),(3,1),(4,1),(4,3)]:
  bd=boot[ai]-boot[bi];delta=rmse[ai]-rmse[bi]
  result["comparisons"][names[ai]+"_minus_"+names[bi]]={"delta_db":delta.tolist(),"ci95_db":np.quantile(bd,[.025,.975],axis=0).T.tolist(),"combined_delta_db":float(delta.mean()),"combined_ci95_db":np.quantile(bd.mean(1),[.025,.975]).tolist()}
 result["seconds"]=time.time()-t0;result["official_promotion_candidate"]=None
 (A/"full_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
 np.savez(A/"paired_per_map.npz",ids=ids,sse=sse,count=count)
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
