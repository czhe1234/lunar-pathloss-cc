"""C147: pinned Sionna RT reflection-depth comparison; no learned calibration."""
import argparse, hashlib, importlib, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c142_backend import rt_environment, load_scene, file_hash, write_json
A=ROOT/"audit/c147_sionna_reflections"
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--mode",choices=["pilot","main"],required=True)
 p.add_argument("--depth",type=int,choices=[1,2,3],required=True)
 p.add_argument("--maps",type=int,nargs="+",required=True)
 p.add_argument("--samples",type=int,required=True)
 p.add_argument("--batch",type=int,default=1048576)
 p.add_argument("--seeds",type=int,nargs="+",default=[13])
 p.add_argument("--bands",type=int,nargs="+",default=[0,1])
 p.add_argument("--refraction",action="store_true")
 args=p.parse_args()
 assert args.samples%args.batch==0
 mi,dr,rt,device=rt_environment("cuda")
 config=json.loads((A/"setup.json").read_text())
 module=importlib.import_module("sionna.rt.radio_map_solvers.radio_map_solver")
 original=module.fibonacci_lattice
 out=A/args.mode/("with_transmission" if args.refraction else "reflection_only")/f"depth{args.depth}"
 out.mkdir(parents=True,exist_ok=True)
 B=args.samples//args.batch
 t0=time.monotonic()
 for mid in args.maps:
  rec=next(x for x in config["records"] if x["map"]==mid)
  scene_dir=ROOT/rec["scene_dir"]
  for name,digest in rec["scene_sha256"].items():assert file_hash(scene_dir/name)==digest
  scene,rx=load_scene(mi,rt,scene_dir,np.array(rec["tx"],np.float32))
  material=scene.objects["terrain"].radio_material
  for band in args.bands:
   freq=(415e6,5.775e9)[band]; sigma=(.0003562,.009324)[band]
   scene.frequency=freq;material.conductivity=sigma
   for seed in args.seeds:
    dest=out/f"{mid}_b{band}_s{seed}.npz"
    meta=dest.with_suffix(".json")
    identity={"map":mid,"band":band,"frequency":freq,"depth":args.depth,
     "refraction":args.refraction,"samples":args.samples,"batch":args.batch,
     "seed":seed,"scene_sha256":rec["scene_sha256"],"tx":rec["tx"],
     "versions":[rt.__version__,mi.__version__,dr.__version__],
     "solver_sha256":file_hash(ROOT/".rt_packages/sionna/rt/radio_map_solvers/radio_map_solver.py")}
    if meta.exists():
     old=json.loads(meta.read_text())
     assert old["identity"]==identity and old["sha256"]==file_hash(dest)
     continue
    checkpoint=dest.with_suffix(".partial.npz")
    total=np.zeros(rx.face_count(),np.float64);half=np.zeros((2,rx.face_count()),np.float64);completed=0
    if checkpoint.exists():
     with np.load(checkpoint) as z:
      assert json.loads(str(z["identity"]))==identity
      total=z["total"];half=z["half"];completed=int(z["completed"])
    solver=rt.RadioMapSolver()
    start=time.monotonic()
    for chunk in range(completed,B):
     def lattice(n):
      assert n==args.batch
      index=dr.arange(mi.UInt,n)*B+chunk
      v=mi.Float64(index)
      u=v/((1+np.sqrt(5))/2)+(seed*.6180339887498949)%1
      u=u-dr.floor(u)
      return mi.Point2f(u,(v+.5)/args.samples)
     module.fibonacci_lattice=lattice
     result=solver(scene,measurement_surface=rx,samples_per_tx=args.batch,
       max_depth=args.depth,los=True,specular_reflection=True,diffuse_reflection=True,
       refraction=args.refraction,seed=seed+7919*chunk,rr_depth=-1,stop_threshold=None)
     power=result.path_gain.numpy()[0].astype(np.float64)/B
     assert power.shape==total.shape and np.isfinite(power).all() and (power>=0).all()
     total+=power;half[chunk%2]+=power
     temp=checkpoint.with_suffix(".tmp.npz")
     np.savez_compressed(temp,total=total,half=half,completed=chunk+1,identity=json.dumps(identity,sort_keys=True))
     temp.replace(checkpoint)
     progress={**identity,"gpu_uuid":device,"completed_chunks":chunk+1,"chunks":B,
               "job_seconds":time.monotonic()-start,"process_seconds":time.monotonic()-t0,
               "pid":os.getpid(),"path":str(dest.relative_to(ROOT))}
     write_json(out/"progress.json",progress)
     print(json.dumps({"event":"chunk",**progress}),flush=True)
    np.savez_compressed(dest,path_gain=total,half_estimates=half*2)
    write_json(meta,{"identity":identity,"sha256":file_hash(dest),"gpu_uuid":device,
                    "seconds":time.monotonic()-start,"pid":os.getpid(),"labels_used":False})
    checkpoint.unlink(missing_ok=True)
    print(json.dumps({"event":"complete","file":str(dest),"seconds":time.monotonic()-start}),flush=True)
    del result,solver
  del scene,rx,material
  dr.flush_malloc_cache()
 module.fibonacci_lattice=original
if __name__=="__main__":main()
