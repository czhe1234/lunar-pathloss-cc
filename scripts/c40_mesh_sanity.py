import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,time,numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import drjit as dr
dr.set_thread_count(8)
import sionna.rt as rt
R=Path(__file__).resolve().parents[1];O=R/'audit/c40_rt_pilot_20261101';D=O/'map0_floor_1024x4000000'
mesh=mi.load_dict({'type':'ply','filename':str(D/'receiver.ply'),'face_normals':True})
scene=rt.load_scene();scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V');scene.rx_array=scene.tx_array;scene.frequency=415e6;scene.add(rt.Transmitter('tx',position=[128,128,5]))
rm=rt.MeshRadioMap(scene,mesh);c=rm.cell_centers.numpy().T
yy,xx=np.mgrid[:256,:256];base=(yy[:-1,:-1]*256+xx[:-1,:-1]).reshape(-1)
faces=np.stack([np.stack([base,base+1,base+257],1),np.stack([base,base+257,base+256],1)],1).reshape(-1,3)
expect=np.stack([xx,yy],-1).reshape(-1,2)[faces].mean(1)
centroid_error=float(np.max(abs(c[:,:2]-expect)))
assert centroid_error<1e-4,(c.shape,centroid_error)
# Flatten only the measurement mesh into z=1; no terrain surface needed for LOS calibration.
params=mi.traverse(mesh);pos=params['vertex_positions'].numpy().reshape(-1,3);pos[:,2]=1;params['vertex_positions']=pos.reshape(-1);params.update()
solver=rt.RadioMapSolver()
powers=[]
for seed in range(16):
 out=solver(scene,measurement_surface=mesh,samples_per_tx=4000000,max_depth=0,los=True,specular_reflection=False,diffuse_reflection=False,refraction=False,seed=20261220+seed)
 powers.append(out.path_gain.numpy()[0])
p=np.mean(powers,axis=0);c=out.cell_centers.numpy().T;dist=np.sqrt((c[:,0]-128)**2+(c[:,1]-128)**2+16)
use=(dist>10)&(dist<50)&(p>0)
e=-10*np.log10(p[use])-20*np.log10(4*np.pi*dist[use]*415e6/299792458.)
rec={'centroid_order_max_xy_difference_m':centroid_error,'flat_mesh_FSPL_RMSE_db':float(np.sqrt(np.mean(e**2))),'bias_db':float(e.mean()),'rays':64000000,'domain':'10m<range<50m; same1m triangular measurement cells'}
(O/'mesh_sanity.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
