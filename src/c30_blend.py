"""C30 cross-fitted convex blending of frozen, independently generated predictions."""
from pathlib import Path
import json,csv,time
import numpy as np
from scipy.optimize import minimize
R=Path(__file__).resolve().parents[1];D=R.parent/"data";A=R/"audit/c30_complementarity_20261101"
def solve(cov):
 f=lambda w:float(w@cov@w)
 r=minimize(f,np.full(3,1/3),jac=lambda w:2*cov@w,bounds=[(0,1)]*3,constraints={"type":"eq","fun":lambda w:w.sum()-1,"jac":lambda w:np.ones(3)},method="SLSQP",options={"ftol":1e-12,"maxiter":200})
 assert r.success,r.message
 return r.x
def main():
 t0=time.time();base=R/"runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh"
 ids=np.array(json.loads((base/"indices.json").read_text())["index_rows"])
 b=[np.load(base/("pl"+band+".npy"),mmap_mode="r") for band in ["415","58"]]
 nets=[]
 for dest in [R/"runs/c21r1_pair_s20261123/dev_B",R/"runs/c22_full_low/dev_complete"]:
  assert json.loads((dest/"ids.json").read_text())==ids.tolist()
  nets.append(np.load(dest/"pred.npy",mmap_mode="r"))
 Y=[np.load(D/f"train_pl{band}.npy",mmap_mode="r") for band in ["415","58"]]
 M=[np.load(D/f"train_mask{band}.npy",mmap_mode="r") for band in ["415","58"]]
 Ph=np.load(R/"features/c21_phase/train.npy",mmap_mode="r")
 rows=list(csv.DictReader((D/"train_index.csv").open()));terrains=sorted(set(rows[int(i)]["terrain_id"] for i in ids))
 perm=np.random.default_rng(20261204).permutation(terrains);half=set(perm[:len(perm)//2])
 fold=np.array([int(rows[int(i)]["terrain_id"] in half) for i in ids]);cov=np.zeros((len(ids),2,3,3,3));count=np.zeros((len(ids),2,3));oracle=np.zeros((len(ids),2))
 bound=np.arcsinh(.25/2)/3
 for j,i in enumerate(ids):
  mx=np.asarray(Ph[i,:8],np.float32).max(0);zone=np.where(mx < -bound,0,np.where(mx>bound,2,1))
  for k in [0,1]:
   mask=np.unpackbits(M[k][i],bitorder="big").reshape(256,256).astype(bool)
   y=Y[k][i].astype(np.float64)
   e=np.stack([b[k][j].astype(np.float64)-y]+[n[j,k].astype(np.float64)-y for n in nets])
   em=e[:,mask];mn=em.min(0);ma=em.max(0)
   oracle[j,k]=np.where((mn<=0)&(ma>=0),0.,np.square(np.abs(em).min(0))).sum()
   for z in range(3):
    m=mask&(zone==z);v=e[:,m];cov[j,k,z]=v@v.T;count[j,k,z]=m.sum()
  if (j+1)%512==0:print(json.dumps({"maps":j+1,"seconds":time.time()-t0}),flush=True)
 totals=count.sum((0,2));base_ss=np.diagonal(cov,axis1=-2,axis2=-1).sum(2)
 result={"scope":"Complete DEV2562; cross-fitting only for blend weights; prior model selection used DEV","names":["C10","C21_B","C22_low"],"models":{n:np.sqrt(base_ss[:,:,k].sum(0)/totals).tolist() for k,n in enumerate(["C10","C21_B","C22_low"])},"blends":{}}
 results={}
 for method in ["global","zones"]:
  sse=np.zeros((len(ids),2));weights={}
  for f in [0,1]:
   tr=fold!=f;va=fold==f;w=np.zeros((2,3,3))
   for k in [0,1]:
    if method=="global":w[k]=solve(cov[tr,k].sum((0,1))/count[tr,k].sum())[None]
    else:
     for z in range(3):w[k,z]=solve(cov[tr,k,z].sum(0)/count[tr,k,z].sum())
   for j in np.flatnonzero(va):
    for k in [0,1]:sse[j,k]=sum(w[k,z]@cov[j,k,z]@w[k,z] for z in range(3))
   weights[str(f)]=w.tolist()
  rms=np.sqrt(sse.sum(0)/totals);baseline=np.array(result["models"]["C10"])
  result["blends"][method]={"rmse":rms.tolist(),"score":float(rms.mean()),"relative_gains":(1-rms/baseline).tolist(),"crossfit_weights":weights,"large_gain_gate":bool(np.all(rms<=baseline*.9))}
  results[method]=sse
 result["label_oracle_convex_hull_rmse"]=np.sqrt(oracle.sum(0)/totals).tolist()
 result["oracle_warning"]="Uses true labels to choose an arbitrary convex combination per pixel; not deployable and not a raw-input information bound."
 result["seconds"]=time.time()-t0
 (A/"results.json").write_text(json.dumps(result,indent=2)+"\n")
 np.savez(A/"per_map.npz",ids=ids,covariance=cov,count=count,fold=fold,oracle=oracle,**results)
 print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
