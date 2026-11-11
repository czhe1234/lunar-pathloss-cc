"""C141 H/TX-only direct-area and receiver-measurement replay on frozen query points.
All computation uses the pinned LLVM CPU backend, two threads, isolated PLY paths.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,json,csv,time,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/"src"))
from c95_alignment import geometry,variant_solver
from c86_targeted import indices,meshfile,mi,rt,dr
from c66_lattice import Mesh
A=R/"audit/c141_direct_receiver"
N=4_000_000_000
CONST=(4*np.pi*415e6/299792458.)**2
def quadrature(n):
 p=[]
 for i in range(n):
  for j in range(n-i):
   for t in ((1/3,2/3) if i+j+1<n else (1/3,)):
    u=(i+t)/n;v=(j+t)/n;p.append([1-u-v,u,v])
 return np.array(p,float)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--limit",type=int,default=0);ap.add_argument("--area-only",action="store_true")
 ap.add_argument("--modes",nargs="+",default=["native_full","native_first"])
 ap.add_argument("--panel",default="input_panel.json");ap.add_argument("--out",default="fit_replay");args=ap.parse_args()
 dr.set_thread_count(2)
 assert mi.variant()=="llvm_ad_mono_polarized" and rt.__version__=="1.1.0"
 panel=json.loads((A/args.panel).read_text())["records"]
 if args.limit:panel=panel[:args.limit]
 rows=list(csv.DictReader((R.parent/"data/train_index.csv").open()));hm=np.load(R.parent/"data/train_hm.npy",mmap_mode="r")
 mesh=Mesh(1,0);lookup=mesh.neighbors.copy();weights=mesh.weights.copy()
 # C135 Transport's exact unsupported-corner fallback, constructed without labels.
 from scipy.spatial import cKDTree
 axis=np.linspace(1/3,255-1/3,256);qx,qy=np.meshgrid(axis,axis)
 xy=np.stack(np.meshgrid(np.arange(256),np.arange(256)),axis=-1).reshape(-1,2)
 unsupported=weights.sum(1)<=.5
 _,near=cKDTree(xy[mesh.faces].mean(1)).query(np.column_stack((qx.ravel()[unsupported],qy.ravel()[unsupported])))
 lookup[unsupported]=near[:,None];weights[unsupported]=[1,0,0]
 weights=np.maximum(weights,0);weights/=weights.sum(1,keepdims=True)
 out=A/args.out;out.mkdir(exist_ok=True);work=A/"work"/f"pid{os.getpid()}";work.mkdir(parents=True,exist_ok=False)
 templates={};t0=time.monotonic()
 for rec in panel:
  mid=rec["map"];dest=out/f"{mid}.npz"
  if dest.exists():continue
  px=np.asarray(rec["pixels"],np.int32);row=rows[mid];h=hm[int(row["hm_row"])]
  vert,rv,fg,fr,tx,_=geometry(h,row,{},templates)
  meshfile(work/"ground.ply",vert,fg);meshfile(work/"rx.ply",rv,fr)
  ground=mi.load_dict({"type":"ply","filename":str(work/"ground.ply"),"face_normals":False,"bsdf":{"type":"holder-material"}})
  mat=rt.RadioMaterial("regolith",relative_permittivity=3.411,conductivity=.0003562,thickness=10.,scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern="lambertian")
  scene=rt.load_scene();scene.edit(add=rt.SceneObject(mi_mesh=ground,name="terrain",radio_material=mat))
  scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern="iso",polarization="V")
  scene.rx_array=scene.tx_array;scene.add(rt.Transmitter("tx",position=tx.tolist()));scene.frequency=415e6
  rx=mi.load_dict({"type":"ply","filename":str(work/"rx.ply"),"face_normals":True})
  ids=lookup[px];w=weights[px];unique,inv=np.unique(ids,return_inverse=True);inv=inv.reshape(ids.shape)
  vv=rv[fr[unique]].astype(float)
  result={"pixels":px,"ids":ids,"weights":w,"tx":tx,"native_vertices":vv,"unique_faces":unique}
  for order in [2,4,8,16]:
   points=np.einsum("qv,fvc->fqc",quadrature(order),vv).reshape(-1,3)
   delta=points-tx;dist=np.linalg.norm(delta,axis=1);direction=delta/dist[:,None]
   ray=mi.Ray3f(o=mi.Point3f(*map(float,tx)),d=mi.Vector3f(direction.T.astype(np.float32)))
   ray.maxt=mi.Float((dist-.0008).astype(np.float32))
   visible=~scene.mi_scene.ray_test(ray).numpy()
   values=(visible/np.maximum(dist**2,.0625)).reshape(len(unique),order*order).mean(1)
   los=visible.reshape(len(unique),order*order).mean(1)
   result[f"area{order*order}_native"]=values[inv]
   result[f"area{order*order}"]=(values[inv]*w).sum(1)
   result[f"area{order*order}_los"]=(los[inv]*w).sum(1)
  print(json.dumps({"event":"area_done","map":mid,"points":len(px),"seconds":time.monotonic()-t0}),flush=True)
  if not args.area_only:
   selected=indices(vv-tx,1.35,N,0.)
   result["selected_rays"]=len(selected)
   for name in args.modes:
    cfg={"native_full":{},"native_first":{"step":"first"},"native_along":{"step":"along"}}[name]
    solver,module=variant_solver(cfg);power=np.zeros(len(fr),float)
    for start in range(0,len(selected),500000):
     rays=selected[start:start+500000];M=len(rays)
     def lattice(n):
      assert n==M
      ns=mi.Float64(mi.UInt(rays));gold=(1+dr.sqrt(mi.Float64(5.)))/2
      u=ns/gold;u-=dr.floor(u);return mi.Point2f(u,ns/(N-1))
     module.fibonacci_lattice=lattice
     res=solver(scene,measurement_surface=rx,samples_per_tx=M,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=13)
     power+=res.path_gain.numpy()[0].astype(float)*(M/N)*CONST
    result[name+"_native"]=power[ids];result[name]=(power[ids]*w).sum(1)
    print(json.dumps({"event":"replay_done","map":mid,"mode":name,"selected_rays":len(selected),"seconds":time.monotonic()-t0}),flush=True)
  tmp=dest.with_suffix(".partial.npz");np.savez_compressed(tmp,**result);tmp.replace(dest)
  if not args.area_only:del solver,res
  del scene,ground,rx
  dr.flush_malloc_cache()
 (out/"complete.json").write_text(json.dumps({"maps":len(panel),"seconds":time.monotonic()-t0,"backend":mi.variant(),"threads":2,"args":vars(args),"labels_used":False,"mask_used":False},indent=2)+"\n")
if __name__=="__main__":main()
