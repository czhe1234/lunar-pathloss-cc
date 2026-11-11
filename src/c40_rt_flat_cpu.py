"""C40 bounded Sionna-RT pilot. Independent scene reconstruction from released H only."""
import sys,os,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
os.environ['CUDA_VISIBLE_DEVICES']=''
info={'backend':'LLVM CPU','threads':8,'reason':'Host lacks libnvoptix.so.1; no driver changes'}
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import drjit as dr
dr.set_thread_count(8)
import sionna.rt as rt
assert mi.variant()=='llvm_ad_mono_polarized',mi.variant()
O=R/'audit/c40_rt_pilot_20261101';O.mkdir(exist_ok=True)
def dump(name,obj):
 (O/name).write_text(json.dumps(obj,indent=2)+'\n')
scene=rt.load_scene()
scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
scene.rx_array=scene.tx_array
scene.add(rt.Transmitter('tx',position=[0,0,5]))
solver=rt.RadioMapSolver()
records=[]
for f in [415e6,5.775e9]:
 scene.frequency=f;t=time.time()
 rm=solver(scene,center=[0,0,1],orientation=[0,0,0],size=[64,64],cell_size=[2,2],samples_per_tx=1000000,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=20261208)
 pg=rm.path_gain.numpy()[0]
 c=rm.cell_centers.numpy()
 np.savez(O/f'flat_{int(f)}.npz',power=pg,centers=c)
 print('flat_shapes',pg.shape,c.shape,flush=True)
 if c.shape[0]==3:c=np.moveaxis(c,0,-1)
 dist=np.sqrt(c[...,0]**2+c[...,1]**2+(c[...,2]-5)**2)
 pl=-10*np.log10(np.maximum(pg,1e-30));fs=20*np.log10(4*np.pi*dist*f/299792458.)
 valid=(dist>10)&(pg>0)
 err=pl[valid]-fs[valid]
 rec={'freq':f,'seconds':time.time()-t,'cells':int(pg.size),'nonzero':int((pg>0).sum()),'FSPL_center_RMSE_db':float(np.sqrt(np.mean(err**2))),'mean_error_db':float(np.mean(err)),'p95_abs_error_db':float(np.percentile(abs(err),95))}
 records.append(rec);print(json.dumps(rec),flush=True)
dump('flat_sanity.json',{'gpu':info,'sionna_rt':rt.__version__,'records':records,'scope':'LOS-only sanity; excludes near cells, point FSPL compared to finite-cell Monte Carlo average'})
