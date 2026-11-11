"""C140: stable nonnegative Neumann sum with receiver-domain error bound.
CPU only. Does not reinterpret cached zero paths as physical zero truth.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2";os.environ["OPENBLAS_NUM_THREADS"]="1"
import json,csv,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,eye
from scipy.sparse.linalg import spsolve
from scipy.special import logsumexp
R=Path(__file__).resolve().parents[1];A=R/"audit/c140_consistency"
plan=json.loads((R/"audit/c136_native/protocol.json").read_text())
rows=list(csv.DictReader((R.parent/"data/train_index.csv").open()))
factor=(299792458./np.array([415e6,5.775e9])/(4*np.pi))**2
def dump(n,x):
 p=A/n;t=p.with_suffix(".tmp");t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def db(g):
 p=np.full_like(g,np.nan);pos=g>0;p[pos]=-10*np.log10(g[pos]);return p
def stats(pred,y,w):
 finite=np.isfinite(pred)&np.isfinite(y);ww=w*finite
 e=np.where(finite,np.nan_to_num(pred)-np.nan_to_num(y),0)
 return {"RMSE":np.sqrt((ww*e**2).sum(0)/ww.sum(0)).tolist(),"valid_weight":ww.sum(0).tolist(),
         "excluded_weight":(w*(~finite)).sum(0).tolist()}
def logpos(x):
 v=np.full_like(x,-np.inf);m=x>0;v[m]=np.log(x[m]);return v
def logmv(edge,x):
 src,dst,lk=edge;v=lk+x[src];ok=np.isfinite(v)
 maximum=np.full(2048,-np.inf);np.maximum.at(maximum,dst[ok],v[ok])
 total=np.zeros(2048);np.add.at(total,dst[ok],np.exp(v[ok]-maximum[dst[ok]]))
 nz=total>0;maximum[nz]+=np.log(total[nz]);return maximum
graphs={};allinfo=[];outputs={}
for split in ["fit","monitor"]:
 ids=plan[split];folder=R/"audit/c101_residual"/split
 d=json.loads((folder/"done.json").read_text());pp=d["points_per_map"];ix={m:i for i,m in enumerate(d["ids"])}
 target=np.load(folder/"target.npy",mmap_mode="r").reshape(-1,pp,4)
 weight=np.load(folder/"weight.npy",mmap_mode="r").reshape(-1,pp)
 pred=[];pred3=[];predglobal=[];ys=[];ws=[];t0=time.monotonic()
 for j,mid in enumerate(ids):
  terrain=rows[mid]["terrain_id"]
  if terrain not in graphs:
   z=np.load(R/f"audit/c132_transport/cache/{terrain}/graph.npz")
   src=z["src"];dst=z["dst"];k=z["k"].astype(float)
   mats=[csr_matrix((k[:,b],(dst,src)),shape=(2048,2048)) for b in range(2)]
   kappa=np.array([np.asarray(m.sum(0)).max() for m in mats])
   adjacency=[(m>0).astype(np.int32) for m in mats]
   edges=[(src[k[:,b]>0],dst[k[:,b]>0],np.log(k[k[:,b]>0,b])) for b in range(2)]
   graphs[terrain]=(mats,kappa,np.tile(z["area"].astype(float),2),adjacency,edges)
  mats,kappa,area,adjacency,edges=graphs[terrain]
  z=np.load(R/f"audit/c132_transport/cache/{terrain}/{mid}.npz")
  Q=z["sq"].astype(float);ri=z["ri"];aw=z["aw"].astype(float);ds=z["ds"].astype(float)
  a=aw/area[ri]
  # Sum >= max aggregated node coefficient, giving a conservative operator bound.
  a_bound=a.sum(1)[:,None]
  direct=ds[:,:2]+ds[:,2:]+(z["rq"].astype(float)*aw[...,None]).sum(1)
  # RX uses native rq for order zero; extra state starts at KQ, not Q.
  reach=np.stack([(adjacency[b]@(Q[:,b]>0).astype(np.int32))>0 for b in range(2)],1)
  for rb in range(2049):
   new=np.stack([(adjacency[b]@reach[:,b].astype(np.int32))>0 for b in range(2)],1)|reach
   if np.array_equal(new,reach):break
   reach=new
  assert rb<2048
  reachable=(direct>0)|((reach[ri] & (aw[...,None]>0)).any(1))
  logterm=logpos(Q);logextra=np.full_like(Q,-np.inf)
  logdirect=logpos(direct);loga=logpos(a);logabound=logpos(a_bound)
  maxdb=np.full(2,np.inf);logthree=None;logglobal=None
  logqnorm=logsumexp(logpos(Q),axis=0)
  for step in range(1,2049):
   logterm=np.stack([logmv(edges[b],logterm[:,b]) for b in range(2)],1)
   logextra=np.logaddexp(logextra,logterm)
   loggain=np.logaddexp(logdirect,logsumexp(logextra[ri]+loga[...,None],axis=1))
   nextlog=np.stack([logmv(edges[b],logterm[:,b]) for b in range(2)],1)
   logtail=logsumexp(nextlog,axis=0)-np.log1p(-kappa)
   logrxbound=logabound+logtail
   err=np.zeros_like(loggain)
   nz=reachable&np.isfinite(loggain)
   err[nz]=(10/np.log(10))*np.logaddexp(0,logrxbound[nz]-loggain[nz])
   err[reachable&(~np.isfinite(loggain))]=np.inf
   maxdb=np.max(err,0)
   if step==3:logthree=loggain.copy()
   relative_global_bound=logtail-logqnorm
   if step>=3 and logglobal is None and np.all(relative_global_bound<=np.log(1e-8)):logglobal=loggain.copy()
   if step>=3 and np.max(maxdb)<=1e-5:break
  assert step<2048,(mid,step,maxdb)
  assert np.array_equal(np.isfinite(loggain),reachable)
  item={"map":mid,"split":split,"iterations":step,"reachable_nodes":reach.sum(0).tolist(),
   "cached_operator_zero_receivers":(~reachable).sum(0).tolist(),
   "max_RX_dB_error_upper_bound":maxdb.tolist()}
  if j<4:
   # Independent sparse direct solve for E=KQ+KE avoids cancellation of L-Q.
   b0=np.stack([mats[b]@Q[:,b] for b in range(2)],1)
   sol=np.stack([spsolve(eye(2048,format="csr")-mats[b],b0[:,b]) for b in range(2)],1)
   assert sol.min()>=-1e-30,(mid,sol.min())
   ref=direct+(np.maximum(sol,0)[ri]*a[...,None]).sum(1)
   reference_pl=db(ref)
   log_pl=-10/np.log(10)*loggain
   pos=(ref>0)&np.isfinite(log_pl)
   differences=np.where(pos,np.abs(reference_pl-log_pl),0)
   mx=np.nanmax(differences,0)
   item["sparse_solve_max_dB_difference_positive_reference"]=mx.tolist()
   unmatched=(ref<=0)&np.isfinite(loggain)
   item["sparse_solve_underflow_count"]=unmatched.sum(0).tolist()
   assert (mx<=maxdb+1e-7).all(),item
   assert (loggain[unmatched]<np.log(np.nextafter(0.,1.))+5).all(),item
  if logglobal is None:logglobal=loggain.copy()
  allinfo.append(item);pred.append(-10/np.log(10)*(loggain+np.log(factor)))
  pred3.append(-10/np.log(10)*(logthree+np.log(factor)))
  predglobal.append(-10/np.log(10)*(logglobal+np.log(factor)))
  lab=target[ix[mid]].astype(float);ys.append(lab[:,:2]);ws.append(weight[ix[mid]].astype(float)[:,None]*lab[:,2:])
  if (j+1)%100==0:print(json.dumps({"split":split,"maps":j+1,"seconds":time.monotonic()-t0,"last_steps":step}),flush=True)
 outputs[split]={"PL":np.concatenate(pred),"PL3":np.concatenate(pred3),"PLglobal":np.concatenate(predglobal),"y":np.concatenate(ys),"w":np.concatenate(ws)}
 np.savez_compressed(A/f"c132_receiver_converged_{split}.npz",ids=ids,PL=outputs[split]["PL"],PL3=outputs[split]["PL3"],PLglobal=outputs[split]["PLglobal"])
 dump("receiver_convergence_progress.json",{"split":split,"maps_done":len(allinfo)})
d=outputs["fit"];p=d["PL"];valid=np.isfinite(p);w=d["w"]*valid
offset=(w*np.where(valid,d["y"]-np.nan_to_num(p),0)).sum(0)/w.sum(0)
summary={"status":"complete","tolerance_max_receiver_dB":1e-5,
 "iterations_min_median_max":np.quantile([x["iterations"] for x in allinfo],[0,.5,1]).tolist(),
 "max_receiver_error_bound":np.max([x["max_RX_dB_error_upper_bound"] for x in allinfo],0).tolist(),
 "FIT_only_dB_offset":offset.tolist(),
 "method":"Log-domain nonnegative Neumann extra sum; exact-cached-operator reachability; conservative receiver-domain tail bound; 8-map independent sparse solve.",
 "limits":"Convergence is for fixed coarse cached K. Cached missing support, native geometry, source/RX quadrature and missing paths are not repaired."}
for split,d in outputs.items():
 p=d["PL"];y=d["y"];w=d["w"]
 previous=np.load(A/f"c132_fixed_point_{split}.npz")
 summary[split]={"raw":stats(p,y,w),"FIT_scaled":stats(p+offset,y,w),
 "stable_three_raw":stats(d["PL3"],y,w),
 "stable_three_to_receiver_converged":stats(d["PL3"],p,w),
 "stable_global_L1_to_receiver_converged":stats(d["PLglobal"],p,w),
 "audit_LminusQ_three_vs_stable_three":stats(db(previous["three"]),d["PL3"],w)}
dump("receiver_convergence_maps.json",allinfo);dump("receiver_convergence.json",summary)
print(json.dumps(summary),flush=True)
