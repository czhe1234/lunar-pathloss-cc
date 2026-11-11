"""CPU gradient-noise diagnostic; no parameter updates."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import json,time,numpy as np
import torch
from c21_data import Data
from c21_model_r1 import Model
R=Path(__file__).resolve().parents[1];A=R/"audit/c22_scale_diagnosis_20261101"
torch.set_num_threads(8)
def main():
 t0=time.time();protocol=json.loads((A/"protocol.json").read_text());data=Data()
 n=np.load(R/"audit/c21_execution_20261101/normalization.npz")
 net=Model(torch.from_numpy(n["mean"]),torch.from_numpy(n["std"]),geometry=False)
 ck=torch.load(R/protocol["parent"],map_location="cpu",weights_only=False)
 net.load_state_dict(ck["model"]);del ck;net.train()
 rng=np.random.default_rng(20261127);ids=protocol["fit_panel"][:32];vectors=[];rows=[]
 def gather(a,p):return a.flatten(2).gather(2,p[:,None].expand(-1,2,-1)).transpose(1,2)
 for group in range(4):
  batchids=ids[group*8:(group+1)*8];dat=[data.sample(i) for i in batchids]
  x,tx,fs,y,m=[torch.from_numpy(np.stack([d[k] for d in dat])) for k in range(5)]
  for draw in range(2):
   p=torch.from_numpy(rng.integers(0,65536,(8,2048)))
   net.zero_grad(set_to_none=True);pred=net(x,tx,p,512)*20+gather(fs,p)
   mm=gather(m,p);loss=(((pred-gather(y,p)).square()*mm).sum((0,1))/mm.sum((0,1))).mean()/400
   loss.backward();v=torch.cat([p.grad.detach().flatten().clone() for p in net.parameters() if p.requires_grad and p.grad is not None])
   vectors.append(v);rows.append({"group":group,"draw":draw,"ids":batchids,"loss_db2":float(loss*400),"norm":float(v.norm())})
   print(json.dumps(rows[-1]),flush=True)
 vec=torch.stack(vectors);cos=(vec@vec.T)/(vec.norm(dim=1)[:,None]*vec.norm(dim=1)[None])
 same=[float(cos[2*i,2*i+1]) for i in range(4)]
 across=[float(cos[2*i,2*j]) for i in range(4) for j in range(i+1,4)]
 result={"scope":"CPU FP32,4 disjoint FIT batches of8 maps,2 independent2048-query samples per map; frozen18000 checkpoint",
  "rows":rows,"cosine_matrix":cos.tolist(),"within_same_maps_independent_pixels_cosine":same,
  "between_disjoint_map_batches_cosine":across,"within_mean":float(np.mean(same)),"between_mean":float(np.mean(across)),
  "seconds":time.time()-t0,"no_parameter_updates":True,
  "limits":"Descriptive estimate over32 maps; gradients can disagree in ordinary well-posed optimization. Not proof of irreducible inputs or a causal bottleneck."}
 (A/"gradient_noise.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)
if __name__=="__main__":main()
