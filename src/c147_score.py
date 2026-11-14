"""Independent FP64 scoring of fixed C147 diagnostic pixels."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/"audit/c147_sionna_reflections"
ARMS=("current_solver","sionna_R1","sionna_R2","sionna_R3")
BANDS=("415MHz","5p775GHz")
GROUPS=("all","tail","under_tail","over_tail","non_tail","C82_good","common_positive")
class Scorer:
 def __init__(self):
  self.protocol=json.loads((A/"protocol.json").read_text());self.setup=json.loads((A/"setup.json").read_text())
  self.y=[np.load(ROOT.parent/f"data/train_pl{b}.npy",mmap_mode="r") for b in ("415","58")]
  self.masks=[np.load(ROOT.parent/f"data/train_mask{b}.npy",mmap_mode="r") for b in ("415","58")]
  self.base=[np.load(ROOT/"audit/c88_mesh/base.npy",mmap_mode="r"),np.load(ROOT/"audit/c101_residual/base0of1/prediction.npy",mmap_mode="r")]
  ids=[json.loads((ROOT/"audit/c88_mesh/base_index.json").read_text())["ids"],json.loads((ROOT/"audit/c101_residual/base0of1/indices.json").read_text())]
  self.lookup={int(v):(s,j) for s,x in enumerate(ids) for j,v in enumerate(x)}
 def ready(self,mid):
  n=self.protocol["samples_per_src"];D=self.protocol["maximum_total_interactions"]
  return all((A/"paths"/f"{mid}_b{b}_s{s}_d{D}_n{n}"/"complete.json").exists() for b in (0,1) for s in self.protocol["seeds"])
 def load_map(self,mid):
  target=np.load(A/"targets"/f"{mid}.npz");px=target["pixels"];inv=target["donor_inverse"];w=target["weights"];nf=len(target["unique_faces"])
  s,j=self.lookup[mid];base=self.base[s][j].reshape(2,-1)[:,px]
  y=np.stack([v[mid].ravel()[px] for v in self.y]);valid=np.stack([np.unpackbits(v[mid])[px].astype(bool) for v in self.masks])
  raw=np.load(A/"baseline"/f"{mid}.npz")["path_gain"]
  old=(raw[target["donors"]]*w[:,:,None]).sum(1).T
  replicates=[];ceiling=[];fspl_error=0;path_counts=[]
  for seed in self.protocol["seeds"]:
   powers=[];ceilings=[];counts=[]
   for band in (0,1):
    folder=A/"paths"/f"{mid}_b{band}_s{seed}_d{self.protocol['maximum_total_interactions']}_n{self.protocol['samples_per_src']}"
    parts=[];caps=[];numbers=[]
    for start in range(0,len(target["points"]),self.protocol["receiver_batch"]):
     file=folder/f"batch{start}.npz";z=np.load(file);meta=json.loads(file.with_suffix(".json").read_text())
     assert np.array_equal(z["points"],target["points"][start:start+self.protocol["receiver_batch"]])
     fspl_error=max(fspl_error,meta["direct_FSPL_max_relative_error"])
     parts.append(z["exact_gain"]);caps.append(z["ceiling_gain"]);numbers.append(z["path_count"])
    exact=np.concatenate(parts,1).reshape(4,nf,4).mean(2)
    cap=np.concatenate(caps).reshape(nf,4).mean(1)
    gain=(exact[:,inv]*w[None]).sum(2)
    powers.append(np.cumsum(gain,axis=0)[1:4])
    ceilings.append((cap[inv]*w).sum(1))
    counts.append(np.concatenate(numbers,1).sum(1).tolist())
   replicates.append(np.stack(powers,1));ceiling.append(np.stack(ceilings));path_counts.append(counts)
  reps=np.stack(replicates)
  pooled=reps.mean(0)
  result=np.concatenate((old[None],pooled),0)
  assert result.shape==(4,2,len(px)) and np.isfinite(result).all() and (result>=0).all()
  assert np.all(np.diff(result[1:],axis=0)>=-1e-30)
  return {"map":mid,"pixels":px,"y":y,"valid":valid,"base":base,"gain":result,"replicates":reps,
          "ceiling":np.mean(ceiling,0),"fspl_error":fspl_error,"path_counts":path_counts}
 def score(self,items):
  y=np.concatenate([v["y"] for v in items],1);valid=np.concatenate([v["valid"] for v in items],1)
  base=np.concatenate([v["base"] for v in items],1).astype(float)
  gains=np.concatenate([v["gain"] for v in items],2)
  prediction=-10*np.log10(np.maximum(gains,1e-30));error=prediction-y
  residual=y-base;tail=np.abs(residual)>=5
  common=(gains>0).all(0)
  masks=[valid,valid&tail,valid&(residual>=5),valid&(residual<=-5),valid&~tail,valid&(np.abs(residual)<=1),valid&common]
  out={"maps":[v["map"] for v in items],"pixels":int(y.shape[1]),"models":{},"regions":{}}
  for name,mask in zip(GROUPS,masks):out["regions"][name]=mask.sum(1).tolist()
  for a,arm in enumerate(ARMS):
   groups={}
   for name,mask in zip(GROUPS,masks):
    count=mask.sum(1);ss=np.where(mask,error[a]**2,0).sum(1);signed=np.where(mask,error[a],0).sum(1)
    groups[name]={"count":count.tolist(),"sse":ss.tolist(),"rmse":np.sqrt(ss/np.maximum(count,1)).tolist(),
       "bias":(signed/np.maximum(count,1)).tolist(),"zero_fraction":((mask&(gains[a]<=0)).sum(1)/np.maximum(count,1)).tolist(),
       "error_ge5_fraction":((mask&(np.abs(error[a])>=5)).sum(1)/np.maximum(count,1)).tolist()}
   out["models"][arm]=groups
  out["delta_R1_minus_R3"]={g:(np.array(out["models"]["sionna_R1"][g]["rmse"])-out["models"]["sionna_R3"][g]["rmse"]).tolist() for g in GROUPS}
  out["sampling_seeds"]=[]
  for seed_index,seed in enumerate(self.protocol["seeds"]):
   rep=np.concatenate([v["replicates"][seed_index] for v in items],2)
   pred=-10*np.log10(np.maximum(rep,1e-30));e=pred-y
   record={"seed":seed}
   for group,mask in zip(GROUPS,masks):
    rms=np.sqrt(np.where(mask[None],e**2,0).sum(2)/np.maximum(mask.sum(1),1)[None])
    record[group]={"RMSE_R1_R2_R3":rms.tolist(),"delta_R1_minus_R3":(rms[0]-rms[2]).tolist()}
   out["sampling_seeds"].append(record)
  out["max_direct_FSPL_relative_error"]=max(v["fspl_error"] for v in items)
  ceilings=np.concatenate([v["ceiling"] for v in items],1)
  out["maximum_total_depth_ceiling_power_fraction"]=np.max(ceilings/np.maximum(gains[3],1e-30),axis=1).tolist()
  out["zero_gain_floor"]=1e-30
  return out
