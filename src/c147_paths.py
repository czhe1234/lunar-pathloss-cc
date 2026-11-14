"""C147 Sionna PathSolver. Cumulative depths share the same generated path set."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.c142_backend import rt_environment,load_scene,write_json,file_hash
A=ROOT/"audit/c147_sionna_reflections"
def powers(paths,maximum):
 real,imag=[v.numpy().astype(np.float64) for v in paths.a]
 valid=paths.valid.numpy();interaction=paths.interactions.numpy()
 refs=((interaction==1)|(interaction==2)).sum(0);length=(interaction!=0).sum(0)
 if valid.ndim==3:valid=valid[:,None,:,None,:]
 if refs.ndim==3:refs=refs[:,None,:,None,:];length=length[:,None,:,None,:]
 power=(real*real+imag*imag)*valid
 assert power.shape==real.shape,(power.shape,real.shape)
 assert set(np.unique(interaction))<={0,1,2,4},np.unique(interaction)
 exact=np.stack([np.where(refs==d,power,0).sum((1,2,3,4)) for d in range(4)])
 counts=np.stack([np.count_nonzero((refs==d)&valid,axis=(1,2,3,4)) for d in range(4)])
 direct=np.where(length==0,power,0).sum((1,2,3,4))
 ceiling=np.where((length==maximum)&(refs<=3),power,0).sum((1,2,3,4))
 assert np.isfinite(exact).all() and (exact>=0).all()
 return exact,counts,direct,ceiling
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--pilot",action="store_true");p.add_argument("--maps",nargs="+",type=int,default=[1819])
 p.add_argument("--samples",type=int,default=65536);p.add_argument("--seeds",nargs="+",type=int,default=[13,37])
 p.add_argument("--bands",nargs="+",type=int,default=[0,1]);p.add_argument("--batch",type=int,default=16)
 p.add_argument("--max-depth",type=int,default=8);p.add_argument("--refraction",action="store_true")
 args=p.parse_args()
 mi,dr,rt,device=rt_environment("cuda")
 setup=json.loads((A/"setup.json").read_text())
 if not args.pilot:
  protocol=json.loads((A/"protocol.json").read_text())
  assert args.samples==protocol["samples_per_src"] and args.max_depth==protocol["maximum_total_interactions"] and args.refraction
  for name,digest in protocol["source_sha256"].items():assert file_hash(ROOT/name)==digest,name
 out=A/("path_pilot" if args.pilot else "paths")
 out.mkdir(exist_ok=True)
 for mid in args.maps:
  rec=next(r for r in setup["records"] if r["map"]==mid)
  target_path=A/"targets"/f"{mid}.npz"
  z=np.load(target_path);points=z["points"]
  if args.pilot:points=points[:min(16,len(points))]
  for name,digest in rec["scene_sha256"].items():assert file_hash(ROOT/rec["scene_dir"]/name)==digest
  scene,rx=load_scene(mi,rt,ROOT/rec["scene_dir"],np.array(rec["tx"],np.float32))
  del rx
  material=scene.objects["terrain"].radio_material
  scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern="iso",polarization="VH")
  for band in args.bands:
   freq=setup["bands"][band];scene.frequency=freq;material.conductivity=setup["material"]["conductivity"][band]
   for seed in args.seeds:
    folder=out/f"{mid}_b{band}_s{seed}_d{args.max_depth}_n{args.samples}";folder.mkdir(exist_ok=True)
    identity={"map":mid,"band":band,"seed":seed,"samples":args.samples,"batch":args.batch,
      "max_depth":args.max_depth,"target_sha256":file_hash(target_path),
      "solver_sha256":file_hash(ROOT/".rt_packages/sionna/rt/path_solvers/path_solver.py"),
      "specular":True,"diffuse":True,"refraction":args.refraction,"cumulative_cap_from_same_paths":True,
      "rx_polarization":"VH; sum received polarization powers","frequency":freq,
      "scene_sha256":rec["scene_sha256"],"tx":rec["tx"]}
    old_names=[]
    begin=time.monotonic()
    for start in range(0,len(points),args.batch):
     dest=folder/f"batch{start}.npz";meta=dest.with_suffix(".json")
     if meta.exists():
      d=json.loads(meta.read_text());assert d["identity"]==identity and d["sha256"]==file_hash(dest)
      continue
     for name in old_names:scene.remove(name)
     block=points[start:start+args.batch];old_names=[]
     for j,point in enumerate(block):
      name=f"rx{j}";scene.add(rt.Receiver(name,position=point.tolist()));old_names.append(name)
     limit=args.samples*len(block)*args.max_depth+len(block)
     start_time=time.monotonic()
     paths=rt.PathSolver()(scene,max_depth=args.max_depth,max_num_paths_per_src=limit,
       samples_per_src=args.samples,synthetic_array=True,los=True,specular_reflection=True,
       diffuse_reflection=True,refraction=args.refraction,seed=seed)
     exact,counts,direct,ceiling=powers(paths,args.max_depth)
     # Independent calibration of all direct coefficients against ray visibility and FSPL.
     tx=np.array(rec["tx"]);delta=block-tx;dist=np.linalg.norm(delta,axis=1)
     ray=mi.Ray3f(o=mi.Point3f(*map(float,tx)),d=mi.Vector3f((delta/dist[:,None]).T.astype(np.float32)))
     ray.maxt=mi.Float((dist-.0008).astype(np.float32))
     visible=~scene.mi_scene.ray_test(ray).numpy()
     expected=visible*(299792458/(4*np.pi*freq*dist))**2
     error=np.abs(direct-expected)/np.maximum(expected,1e-30)
     assert np.max(error)<.005, (mid,band,start,float(np.max(error)))
     temp=dest.with_suffix(".tmp.npz")
     np.savez_compressed(temp,exact_gain=exact,path_count=counts,direct_gain=direct,ceiling_gain=ceiling,points=block)
     temp.replace(dest)
     record={"identity":identity,"start":start,"receivers":len(block),"seconds":time.monotonic()-start_time,
      "sha256":file_hash(dest),"GPU":device,"pid":os.getpid(),"labels_used":False,
      "direct_FSPL_max_relative_error":float(error.max()),"candidate_upper_bound":limit}
     write_json(meta,record)
     write_json(out/f"progress_{os.getpid()}.json",{"map":mid,"band":band,"seed":seed,"start":start,
       "points":len(points),"completed":min(len(points),start+args.batch),"GPU":device,"pid":os.getpid(),
       "job_seconds":time.monotonic()-begin})
     print(json.dumps({"event":"batch",**record}),flush=True)
     del paths,exact,counts,direct,ceiling
    for name in old_names:scene.remove(name)
    write_json(folder/"complete.json",{"identity":identity,"points":len(points),"seconds":time.monotonic()-begin})
    print("PATH_GROUP_COMPLETE",mid,band,seed,args.max_depth,time.monotonic()-begin,flush=True)
  del scene,material
  dr.flush_malloc_cache()
if __name__=="__main__":main()
