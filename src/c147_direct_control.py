"""C147 direct-path coverage control. All fields use original Sionna calculations."""
import argparse,json,sys,time,importlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.c142_backend import rt_environment,load_scene,write_json,file_hash
from src.c147_paths import powers
A=ROOT/"audit/c147_sionna_reflections"
def main():
 p=argparse.ArgumentParser();p.add_argument("--maps",nargs="+",type=int);args=p.parse_args()
 mi,dr,rt,device=rt_environment("cuda")
 setup=json.loads((A/"setup.json").read_text())
 mod=importlib.import_module("sionna.rt.path_solvers.sb_candidate_generator")
 original=mod.spawn_ray_from_sources
 out=A/"direct_coverage";out.mkdir(exist_ok=True)
 for rec in setup["records"]:
  mid=rec["map"]
  if args.maps and mid not in args.maps:continue
  target=np.load(A/"targets"/f"{mid}.npz");points=target["points"];tx=np.array(rec["tx"])
  scene,rx=load_scene(mi,rt,ROOT/rec["scene_dir"],tx);del rx
  material=scene.objects["terrain"].radio_material
  scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern="iso",polarization="VH")
  for band in (0,1):
   dest=out/f"{mid}_b{band}.npz"
   if dest.with_suffix(".json").exists():continue
   scene.frequency=setup["bands"][band];material.conductivity=setup["material"]["conductivity"][band]
   result=[];counts=[];transmissions=[];old=[]
   start_time=time.monotonic()
   for start in range(0,len(points),16):
    for name in old:scene.remove(name)
    old=[];block=points[start:start+16]
    for j,pos in enumerate(block):
     name=f"rx{j}";scene.add(rt.Receiver(name,position=pos.tolist()));old.append(name)
    delta=block-tx;direction=delta/np.linalg.norm(delta,axis=1)[:,None]
    def spawn(lattice,size,sources):
     assert size==len(block) and dr.shape(sources)[1]==1
     ray=mi.Ray3f(o=dr.repeat(sources,size),d=mi.Vector3f(direction.T.astype(np.float32)))
     ray.time=dr.zeros(mi.Float,size)
     return ray
    mod.spawn_ray_from_sources=spawn
    paths=rt.PathSolver()(scene,max_depth=32,max_num_paths_per_src=len(block)**2*32+len(block),
     samples_per_src=len(block),synthetic_array=True,los=True,specular_reflection=False,
     diffuse_reflection=False,refraction=True,seed=13)
    exact,number,_,_=powers(paths,32)
    assert np.max(number[0])<=1, ("Duplicate straight path",mid,band,start,number[0].tolist())
    result.append(exact[0]);counts.append(number[0])
    interactions=paths.interactions.numpy();transmissions.append(int(np.max((interactions==4).sum(0))))
    del paths
   for name in old:scene.remove(name)
   np.savez_compressed(dest,gain=np.concatenate(result),path_count=np.concatenate(counts),points=points)
   write_json(dest.with_suffix(".json"),{"map":mid,"band":band,"sha256":file_hash(dest),"seconds":time.monotonic()-start_time,
    "maximum_transmission_count":max(transmissions),"method":"Seed candidate search along exact source-target direction; original Sionna refraction field calculator; no diffuse paths, so no Monte Carlo weighting change.","GPU":device,"labels_used":False})
   print("DIRECT_COVERAGE_DONE",mid,band,max(transmissions),time.monotonic()-start_time,flush=True)
  del scene,material;dr.flush_malloc_cache()
 mod.spawn_ray_from_sources=original
if __name__=="__main__":main()
