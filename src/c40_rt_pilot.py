"""Small TRAIN-only RT reconstruction pilot; CPU LLVM, no GPU or host driver changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,csv,argparse
from pathlib import Path
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import drjit as dr
dr.set_thread_count(8)
import sionna.rt as rt
from scipy.interpolate import LinearNDInterpolator
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
p=argparse.ArgumentParser();p.add_argument('--map',type=int,default=-1);p.add_argument('--batches',type=int,default=4);p.add_argument('--samples',type=int,default=1000000);p.add_argument('--tx-height',choices=['floor','nominal'],default='floor');a=p.parse_args()
rows=list(csv.DictReader((D/'train_index.csv').open()));dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
ids=[i for i,q in enumerate(rows) if q['terrain_id'] not in dev]
idx=ids[0] if a.map<0 else a.map
assert idx in ids
row=rows[idx];h=np.array(np.load(D/'train_hm.npy',mmap_mode='r')[int(row['hm_row'])],np.float32)
tr,tc=int(row['tx_row']),int(row['tx_col']);zt=float(h[tr,tc]);zt=(np.floor(zt) if a.tx_height=='floor' else zt)+3.
O=R/'audit/c40_rt_pilot_20261101'/f'map{idx}_{a.tx_height}_{a.batches}x{a.samples}'
O.mkdir(exist_ok=False)
yy,xx=np.mgrid[:256,:256]
v=np.stack([xx,yy,h],-1).astype('<f4').reshape(-1,3)
base=(yy[:-1,:-1]*256+xx[:-1,:-1]).reshape(-1)
faces=np.stack([np.stack([base,base+1,base+257],1),np.stack([base,base+257,base+256],1)],1).reshape(-1,3).astype('<i4')
centers=v[faces].mean(1)
def ply(path,verts,face):
 dtype=np.dtype([('n','u1'),('v','<i4',(3,))]);f=np.empty(len(face),dtype=dtype);f['n']=3;f['v']=face
 hdr=f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(face)}\nproperty list uchar int vertex_indices\nend_header\n"
 with path.open('wb') as out:out.write(hdr.encode());verts.tofile(out);f.tofile(out)
ply(O/'terrain.ply',v,faces)
rxv=v.copy();rxv[:,2]+=1;ply(O/'receiver.ply',rxv,faces)
mat=rt.RadioMaterial('regolith',relative_permittivity=3.411,conductivity=.0003562,thickness=10.,scattering_coefficient=.632,xpd_coefficient=.1,scattering_pattern='lambertian')
mesh=mi.load_dict({'type':'ply','filename':str(O/'terrain.ply'),'face_normals':True,'bsdf':{'type':'holder-material'}})
obj=rt.SceneObject(mi_mesh=mesh,name='terrain',radio_material=mat)
scene=rt.load_scene();scene.edit(add=obj)
scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V');scene.rx_array=scene.tx_array
scene.add(rt.Transmitter('tx',position=[tc,tr,float(zt)]))
meas=mi.load_dict({'type':'ply','filename':str(O/'receiver.ply'),'face_normals':True})
solver=rt.RadioMapSolver();t0=time.time()
meta={'index':idx,'sample_id':row['sample_id'],'terrain_id':row['terrain_id'],'tx':[tc,tr,float(zt)],'tx_hypothesis':a.tx_height,'vertex_count':len(v),'face_count':len(faces),'sionna':rt.__version__,'backend':'LLVM CPU8','sampling':vars(a),'scope':'Reconstructed 1m mesh, not recovered original mesh; TRAIN only','interactions':{'max_depth':100,'los':True,'specular_reflection':True,'diffuse_reflection':True,'refraction':True}}
(O/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
stats=[]
for k,(f,sigma,band) in enumerate([(415e6,.0003562,'415'),(5.775e9,.009324,'58')]):
 scene.frequency=f;mat.conductivity=sigma
 ys=np.load(D/f'train_pl{band}.npy',mmap_mode='r')[idx].astype(float)
 mask=np.unpackbits(np.load(D/f'train_mask{band}.npy',mmap_mode='r')[idx],bitorder='big').reshape(256,256).astype(bool)
 power=[]
 for b in range(a.batches):
  st=time.time()
  rm=solver(scene,measurement_surface=meas,samples_per_tx=a.samples,max_depth=100,los=True,specular_reflection=True,diffuse_reflection=True,refraction=True,seed=20261209+b)
  pg=rm.path_gain.numpy()[0].astype(np.float64);assert pg.shape==(len(faces),)
  assert np.isfinite(pg).all() and (pg>=0).all()
  power.append(pg);avg=np.mean(power,axis=0)
  np.save(O/f'face_power{band}.npy',avg)
  print(json.dumps({'band':band,'batch':b+1,'seconds':time.time()-st,'nonzero_face_fraction':float((avg>0).mean())}),flush=True)
  if (b+1) in [1,2,4,8,16,32,64,128,256,512,1024] or b==a.batches-1:
   pggrid=LinearNDInterpolator(centers[:,:2],avg)(xx,yy)
   inside=np.isfinite(pggrid);pred=np.clip(-10*np.log10(np.maximum(np.nan_to_num(pggrid),1e-30)),0,180)
   use=mask&inside
   e=pred[use]-ys[use]
   rec={'band':band,'batches':b+1,'RMSE_clamped_all_valid_in_support':float(np.sqrt(np.mean(e**2))),'valid_in_support':int(use.sum()),'nonzero_valid_fraction':float((pggrid[use]>0).mean()),'bias':float(e.mean())}
   stats.append(rec);print(json.dumps(rec),flush=True)
   np.save(O/f'pl{band}.npy',pred.astype(np.float32));np.save(O/f'support{band}.npy',inside)
 if a.batches>=2:
  p1=np.mean(power[::2],axis=0);p2=np.mean(power[1::2],axis=0);non=(p1>0)&(p2>0)
  stats.append({'band':band,'split_half_cell_RMSE_db':float(np.sqrt(np.mean((10*np.log10(p1[non]/p2[non]))**2))),'both_nonzero_fraction':float(non.mean()),'warning':'Half agreement is a Monte Carlo diagnostic, not label agreement.'})
 np.savez(O/f'power_halves{band}.npz',odd=p1,even=p2)
(O/'result.json').write_text(json.dumps({'config':meta,'seconds':time.time()-t0,'records':stats},indent=2)+'\n')
