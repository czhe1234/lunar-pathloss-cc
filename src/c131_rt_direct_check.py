"""Independent Sionna 1.1.0 LoS-only check of C131's fixed candidate direct floor."""
import sys,json,time,csv,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
import numpy as np,torch
from src.c131_direct import Direct
gpu=gpu_guard();torch.set_num_threads(3)
sys.path.insert(0,str(R/'.rt_packages'))
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import drjit as dr
dr.set_thread_count(8)
import sionna.rt as rt
A=R/'audit/c131_transport';O=A/'official_direct';O.mkdir(exist_ok=True)
def meshfile(path,vertices,faces):
 data=np.empty(len(faces),dtype=np.dtype([('n','u1'),('idx','<i4',(3,))]));data['n']=3;data['idx']=faces
 head=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n'
 with path.open('wb') as f:f.write(head.encode());vertices.astype('<f4').tofile(f);data.tofile(f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=8000000);ap.add_argument('--limit',type=int,default=8);ap.add_argument('--solver',choices=['original','line'],default='original');a=ap.parse_args()
 tag=str(a.samples) if a.solver=='original' else str(a.samples)+'_line'
 plan=json.loads((A/'protocol.json').read_text())
 # Choose terrain-stratified cases before looking at their C131 errors.
 rows=list(csv.DictReader((DATA/'train_index.csv').open()));hmap=np.load(DATA/'train_hm.npy',mmap_mode='r')
 grouped={}
 for i in plan['monitor']:grouped.setdefault(rows[i]['terrain_id'],[]).append(i)
 rng=np.random.default_rng(131039);names=sorted(grouped);order=rng.permutation(len(names))
 ids=[grouped[names[j]][0] for j in order[:a.limit]]
 lookup=Direct(16);pixel_ids=lookup.rt.ids.cpu().numpy();pixel_w=lookup.rt.weights.cpu().numpy()
 ya,xa=np.mgrid[:256,:256];axis=np.linspace(-128,128,256,dtype='float32')
 cell=(ya[:-1,:-1]*256+xa[:-1,:-1]).ravel()
 faces=np.concatenate([np.stack([cell,cell+257,cell+1],1),np.stack([cell,cell+256,cell+257],1)])
 vertices=np.stack([axis[xa],-axis[ya],xa*0],-1).astype('float32').reshape(-1,3)
 target=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')]
 valid=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
 from src.c131_sionna_line import LineRadioMapSolver
 solver=rt.RadioMapSolver() if a.solver=='original' else LineRadioMapSolver();records=[];all_stats=[];start=time.monotonic()
 atomic_json(O/f'ids_{tag}.json',ids)
 for k,i in enumerate(ids):
  t=time.monotonic();row=rows[i];h=hmap[int(row['hm_row'])];tr,tc=int(row['tx_row']),int(row['tx_col'])
  vertices[:,2]=h.ravel();rxv=vertices.copy();rxv[:,2]+=1
  meshfile(O/'ground.ply',vertices,faces);meshfile(O/'receiver.ply',rxv,faces)
  mat=rt.RadioMaterial('regolith',relative_permittivity=3.411,conductivity=.0003562,thickness=10.,scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern='lambertian')
  ground=mi.load_dict({'type':'ply','filename':str(O/'ground.ply'),'face_normals':False,'bsdf':{'type':'holder-material'}})
  scene=rt.load_scene();scene.edit(add=rt.SceneObject(mi_mesh=ground,name='terrain',radio_material=mat))
  scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
  scene.rx_array=scene.tx_array;scene.add(rt.Transmitter('tx',position=[tc-128,127-tr,float(np.floor(h[min(tr+1,255),tc])+3)]))
  scene.frequency=415e6;rx=mi.load_dict({'type':'ply','filename':str(O/'receiver.ply'),'face_normals':True})
  rm=solver(scene,measurement_surface=rx,samples_per_tx=a.samples,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=13)
  cell_power=rm.path_gain.numpy()[0].astype('float64')
  phys=(cell_power[pixel_ids]*pixel_w).sum(1).reshape(256,256)
  native=lookup.pixels(h,tr,tc).cpu().numpy()
  ours=native[2].astype('float64')*(299792458/(4*np.pi*415e6))**2
  np.savez(O/f'power_{i}_{tag}.npz',sionna415=phys,native415=ours,cell_power=cell_power)
  y=np.stack([x[i] for x in target]).astype('float64')
  mask=np.stack([np.unpackbits(x[i],bitorder='big').reshape(256,256) for x in valid]).astype(bool)
  fscale=np.array([1,(415e6/5.775e9)**2])[:,None,None]
  bounds=[];ss=[]
  for power in [ours,phys]:
   cap=-10*np.log10(np.maximum(power[None]*fscale,1e-300))
   error=np.maximum(y-cap,0)*mask;ss.append((error**2).sum((1,2)))
   bounds.append(np.sqrt(ss[-1]/mask.sum((1,2))).tolist())
  strong=(native[3]>.999)&(ours>1e-18)&(phys>0)
  delta=-10*np.log10(np.maximum(phys,1e-300)/np.maximum(ours,1e-300))
  rec=dict(map=i,terrain=row['terrain_id'],full_clear_points=int(strong.sum()),
   direct_difference_clear_rms=float(np.sqrt(np.mean(delta[strong]**2))),
   direct_difference_clear_median=float(np.median(delta[strong])),
   floor_native_then_official=bounds,seconds=time.monotonic()-t)
  records.append(rec);all_stats.append(dict(sse=np.array(ss).tolist(),count=mask.sum((1,2)).tolist()))
  atomic_json(O/f'progress_{tag}.json',dict(maps=k+1,total=len(ids),seconds=time.monotonic()-start,records=records))
  print(json.dumps(rec),flush=True)
  del scene,ground,mat,rx,rm
  dr.flush_malloc_cache()
 n=sum(np.array(z['count']) for z in all_stats);s=sum(np.array(z['sse']) for z in all_stats)
 result=dict(status='complete',GPU=gpu,solver=rt.__version__,variant=mi.variant(),maps=len(ids),samples_per_tx=a.samples,
  ids=ids,receiver_spawn=a.solver,records=records,stats=all_stats,aggregate_floor_native_then_official=np.sqrt(s/n).tolist(),
  seconds=time.monotonic()-start,geometry='Same fixed C79/C131 candidate, no geometry search',labels_used='scoring only',official_VAL_used=False)
 atomic_json(O/f'result_{tag}.json',result);print('DONE',json.dumps({k:v for k,v in result.items() if k not in ['records','stats']}),flush=True)
if __name__=='__main__':main()
