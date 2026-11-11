"""Independent FP64 scoring and paired-terrain comparison for C22."""
from pathlib import Path
import csv,json,time
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/"data";A=R/"audit/c22_scale_diagnosis_20261101"
def main():
 t0=time.time();old=np.load(R/"audit/c21_execution_20261101/r1_paired_per_map.npz")
 ids=old["ids"];count=old["count"];rows=list(csv.DictReader((D/"train_index.csv").open()))
 Y=[np.load(D/f"train_pl{b}.npy",mmap_mode="r") for b in ["415","58"]]
 M=[np.load(D/f"train_mask{b}.npy",mmap_mode="r") for b in ["415","58"]]
 names=["C10","C21_A","C22_low","C22_rewarm"];sse=[old["sse"][0],old["sse"][1]];checks={}
 for name,runname in zip(names[2:],["c22_full_low","c22_full_rewarm"]):
  post=json.loads((A/(runname+"_post.json")).read_text());run=R/"runs"/post["chosen"]/"dev_complete"
  assert json.loads((run/"ids.json").read_text())==ids.tolist()
  pred=np.load(run/"pred.npy",mmap_mode="r");ss=np.zeros((len(ids),2))
  for j,i in enumerate(ids):
   for b in [0,1]:
    m=np.unpackbits(M[b][i],bitorder="big").reshape(256,256).astype(bool)
    assert int(m.sum())==count[j,b]
    e=pred[j,b].astype(np.float64)-Y[b][i].astype(np.float64)
    ss[j,b]=np.square(e[m]).sum()
  rmse=np.sqrt(ss.sum(0)/count.sum(0))
  diff=float(np.max(np.abs(rmse-np.array(post["complete_DEV"]["rmse"]))))
  assert diff<1e-8,(name,diff)
  checks[name]={"independent_max_difference_db":diff,"selected_run":post["chosen"]};sse.append(ss)
 sse=np.stack(sse);terrains=sorted(set(rows[int(i)]["terrain_id"] for i in ids));lookup={t:k for k,t in enumerate(terrains)}
 ts=np.zeros((len(names),len(terrains),2));tc=np.zeros((len(terrains),2))
 for j,i in enumerate(ids):
  k=lookup[rows[int(i)]["terrain_id"]];ts[:,k]+=sse[:,j];tc[k]+=count[j]
 draws=np.random.default_rng(20261129).integers(0,len(terrains),(2000,len(terrains)))
 bc=tc[draws].sum(1);boot=np.stack([np.sqrt(s[draws].sum(1)/bc) for s in ts])
 rmse=np.sqrt(sse.sum(1)/count.sum(0))
 result={"scope":"CompleteDEV2562,612terrain groups; official formula, not officialVAL","count":count.sum(0).astype(int).tolist(),"models":{},"comparisons":{},"bootstrap":"2000 paired terrain draws; conditional on this seed, not seed uncertainty","independent_checks":checks}
 for k,n in enumerate(names):result["models"][n]={"rmse":rmse[k].tolist(),"score":float(rmse[k].mean())}
 for ai,bi in [(2,0),(3,0),(2,1),(3,1),(3,2)]:
  delta=rmse[ai]-rmse[bi];bd=boot[ai]-boot[bi]
  result["comparisons"][names[ai]+"_minus_"+names[bi]]={"delta_db":delta.tolist(),"ci95_db":np.quantile(bd,[.025,.975],axis=0).T.tolist(),"combined_delta_db":float(delta.mean()),"combined_ci95_db":np.quantile(bd.mean(1),[.025,.975]).tolist()}
 result["official_promotion_candidate"]=None
 for name in names[2:]:
  c=result["comparisons"][name+"_minus_C10"]
  if all(d<0 for d in c["delta_db"]) and all(ci[1]<0 for ci in c["ci95_db"]):result["official_promotion_candidate"]=name
 result["seconds"]=time.time()-t0
 (A/"full_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
 np.savez(A/"paired_per_map.npz",ids=ids,sse=sse,count=count)
 lc=json.loads((R/"runs/c22_full_low/complete.json").read_text());hc=json.loads((R/"runs/c22_full_rewarm/complete.json").read_text())
 assert lc["query_order_sha256"]==hc["query_order_sha256"]
 p=A/"comparability.json";d=json.loads(p.read_text());d["query_order_pending_until_completion"]=False;d["full_pair_query_order_match"]=True;d["query_order_sha256"]=lc["query_order_sha256"];p.write_text(json.dumps(d,indent=2)+"\n")
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
