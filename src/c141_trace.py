"""C141 diagnostic: receiver crossings can change the ray sent to the ground.
Fixed geometry queries only. No label, mask or learned prediction is read here.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import sys,json,csv,time
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/"src"))
from c141_direct_replay import geometry,meshfile,quadrature,mi,rt,dr
A=R/"audit/c141_direct_receiver"
def main():
 dr.set_thread_count(2)
 records=json.loads((A/"input_panel.json").read_text())["records"]
 rows=list(csv.DictReader((R.parent/"data/train_index.csv").open()))
 hm=np.load(R.parent/"data/train_hm.npy",mmap_mode="r")
 work=A/"work"/f"trace{os.getpid()}";work.mkdir(parents=True,exist_ok=False)
 selected={2003:63494,3970:6088,1819:5926};out=[]
 for mid,pixel in selected.items():
  rec=next(r for r in records if r["map"]==mid)
  z=np.load(A/f"fit_replay/{mid}.npz");j=int(np.flatnonzero(z["pixels"]==pixel)[0])
  # One remaining contradiction plus the first ordinary query fixed in input panel.
  # The input panel already orders diagnostic queries before ordinary controls.
  cases=[("remaining_counterexample",j),("ordinary_control",len(z["pixels"])-8)]
  row=rows[mid];h=hm[int(row["hm_row"])]
  v,rv,fg,fr,tx,_=geometry(h,row,{}, {})
  meshfile(work/"ground.ply",v,fg);meshfile(work/"rx.ply",rv,fr)
  ground=mi.load_dict({"type":"ply","filename":str(work/"ground.ply"),"face_normals":True})
  rx=mi.load_dict({"type":"ply","filename":str(work/"rx.ply"),"face_normals":True})
  joint=mi.load_dict({"type":"scene","ground":ground,"receiver":rx})
  pure=mi.load_dict({"type":"scene","ground":ground})
  for group,j in cases:
   ids=z["ids"][j];vv=rv[fr[ids]].astype(float)
   pp=np.einsum("qv,fvc->fqc",quadrature(4),vv).reshape(-1,3)
   direction=(pp-tx);direction/=np.linalg.norm(direction,axis=1,keepdims=True)
   ray=mi.Ray3f(mi.Point3f(*map(float,tx)),mi.Vector3f(direction.T.astype(np.float32)))
   N=len(pp);g0=pure.ray_intersect(ray)
   purevalid=g0.is_valid().numpy();pureid=g0.prim_index.numpy();purep=g0.p.numpy().T
   active=np.ones(N,bool);visits=np.zeros(N,int);target_hit=np.zeros(N,bool)
   target_face=np.repeat(ids,16);max_shift=np.zeros(N)
   groundvalid=np.zeros(N,bool);groundid=np.full(N,-1,int);groundp=np.full((N,3),np.nan)
   escapes=np.zeros(N,bool);traces=[[] for _ in range(N)]
   for step in range(128):
    si=joint.ray_intersect(ray,active=mi.Bool(active));valid=si.is_valid().numpy()&active
    isrx=(si.shape==rx).numpy()&valid;isground=valid&~isrx
    pos=si.p.numpy().T;fid=si.prim_index.numpy()
    for k in np.flatnonzero(active):
     if not valid[k]:event={"step":step,"event":"escape"}
     else:event={"step":step,"event":"rx" if isrx[k] else "ground","face":int(fid[k]),"position":pos[k].astype(float).tolist()}
     traces[k].append(event)
    escapes|=active&~valid
    groundvalid|=isground;groundid[isground]=fid[isground];groundp[isground]=pos[isground]
    target_hit|=isrx&(fid==target_face)
    visits+=isrx
    if not np.any(isrx):
     active[:]=False;break
    nr=si.spawn_ray(d=ray.d)
    o=nr.o.numpy().T.astype(float);offset=o-tx
    perp=offset-(offset*direction).sum(1)[:,None]*direction
    max_shift=np.maximum(max_shift,np.where(isrx,np.linalg.norm(perp,axis=1),0))
    ray=nr;active=isrx
   both=purevalid&groundvalid
   gpdist=np.linalg.norm(groundp[both]-purep[both],axis=1)
   record={"map":mid,"pixel":int(z["pixels"][j]),"group":group,"rays":N,
    "target_face_hit":int(target_hit.sum()),"ground_only_hits":int(purevalid.sum()),
    "with_receiver_ground_hits":int(groundvalid.sum()),"ground_hit_status_changed":int((purevalid!=groundvalid).sum()),
    "ground_face_changed_among_both":int((pureid[both]!=groundid[both]).sum()),"both_ground_hits":int(both.sum()),
    "ground_position_shift_m_q50_q90_max":np.quantile(gpdist,[.5,.9,1]).tolist() if len(gpdist) else [],
    "ray_transverse_displacement_m_q50_q90_max":np.quantile(max_shift,[.5,.9,1]).tolist(),
    "receiver_hits_q50_q90_max":np.quantile(visits,[.5,.9,1]).tolist(),
    "unfinished_rays":int(active.sum())}
   detailed={"summary":record,"directions":direction.tolist(),"target_faces":target_face.tolist(),"target_hits":target_hit.tolist(),
    "pure_ground_face":np.where(purevalid,pureid,-1).tolist(),"receiver_ground_face":groundid.tolist(),"traces":traces}
   (A/f"trace_{mid}_{int(z['pixels'][j])}.json").write_text(json.dumps(detailed,indent=2,allow_nan=False)+"\n")
   out.append(record);print(json.dumps(record),flush=True)
  del joint,pure,ground,rx
 (A/"trajectory_summary.json").write_text(json.dumps({"labels_read":False,"scope":"fixed FIT geometry queries, not a random-ray prevalence estimate","records":out},indent=2)+"\n")
if __name__=="__main__":main()
