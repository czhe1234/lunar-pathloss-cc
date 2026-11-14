"""C147: reuse old raw power caches; ray trace only the uncached counterexample."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard
from src.c69_registered import RegisteredTransport
from src.c142_backend import file_hash,write_json
import torch
A=ROOT/"audit/c147_sionna_reflections"
def main():
 gpu=gpu_guard();torch.set_num_threads(2)
 setup=json.loads((A/"setup.json").read_text())
 rows=list(csv.DictReader((ROOT.parent/"data/train_index.csv").open()))
 heights=np.load(ROOT.parent/"data/train_hm.npy",mmap_mode="r")
 out=A/"baseline";out.mkdir(exist_ok=True)
 engine=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=128)
 centers=engine.xy[engine.faces].mean(1)
 for rec in setup["records"]:
  mid=rec["map"];dest=out/f"{mid}.npz"
  if dest.with_suffix(".json").exists():continue
  row=rows[mid];h=heights[int(row["hm_row"])]
  tr,tc=int(row["tx_row"]),int(row["tx_col"])
  if rec["baseline_cache"]:
   ht=torch.as_tensor(np.array(h),device="cuda")
   x=tc*255/256;y=(tr+1)*255/256;z=engine.source_height(h,tr,tc)
   rz=ht.reshape(-1)[engine.faces].mean(1)+1
   d2=((centers[:,0]-x*engine.spacing)**2+(centers[:,1]-y*engine.spacing)**2+(rz-z)**2).clamp_min(.0625)
   values=np.load(ROOT/rec["baseline_cache"])
   native=values[:,:8].astype(np.float64)/d2.cpu().numpy()[:,None]
   source={"cache":rec["baseline_cache"],"sha256":file_hash(ROOT/rec["baseline_cache"]),"new_ray_trace":False}
  else:
   direct,spec,diff,*_=engine.raw(h,tr,tc)
   native=torch.cat((direct[:,:2],spec[:,:2],diff[:,:2],direct[:,:2]+spec[:,:2]+diff[:,:2]),1).cpu().numpy().astype(np.float64)
   source={"new_ray_trace":True,"reason":"No prior original128-direction raw cache for 00414_00"}
  factors=(299792458/(4*np.pi*np.array(setup["bands"])))**2
  gains=native*factors[np.arange(8)%2]
  assert np.isfinite(gains).all() and (gains>=0).all()
  np.savez_compressed(dest,path_gain=gains[:,6:8],components=gains)
  write_json(dest.with_suffix(".json"),{"map":mid,"source":source,"sha256":file_hash(dest),
    "material":setup["material"],"gpu":gpu,"labels_used":False,"direct_samples":4,"specular_samples":16,
    "diffuse_directions":128,"max_ground_reflections":1})
  print("baseline_done",mid,source,flush=True)
if __name__=="__main__":main()
