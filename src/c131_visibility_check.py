"""Ground-only Mitsuba visibility vs CC DDA: isolates RM receiver traversal from terrain occlusion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.c131_rt_direct_check import *
def check():
 ids=json.loads((O/'ids_8000000.json').read_text())
 rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 direct=Direct(4);q=direct.q.cpu().numpy()
 yy,xx=np.mgrid[:256,:256];ax=np.linspace(-128,128,256,dtype='float32')
 cell=(yy[:-1,:-1]*256+xx[:-1,:-1]).ravel()
 faces=np.concatenate([np.stack([cell,cell+1,cell+257],1),np.stack([cell,cell+257,cell+256],1)])
 vertices=np.stack([ax[xx],-ax[yy],xx*0],-1).astype('float32').reshape(-1,3)
 ix=direct.rt.ids.cpu().numpy();ww=direct.rt.weights.cpu().numpy();records=[]
 for i in ids:
  r=rows[i];h=hm[int(r['hm_row'])];tr,tc=int(r['tx_row']),int(r['tx_col'])
  vertices[:,2]=h.ravel();rv=vertices.copy();rv[:,2]+=1
  meshfile(O/'visibility_ground.ply',vertices,faces[:,[0,2,1]])
  scene=mi.load_dict({'type':'scene','terrain':{'type':'ply','filename':str(O/'visibility_ground.ply'),'face_normals':True}})
  tx=np.array([tc-128,127-tr,float(np.floor(h[min(tr+1,255),tc])+3)],dtype='float32')
  xyz=np.einsum('qj,fjd->fqd',q,rv[faces]).reshape(-1,3)
  dest=mi.Point3f(xyz.T.copy());origin=mi.Point3f(tx.tolist());dv=dest-origin;length=dr.norm(dv)
  ray=mi.Ray3f();ray.o=origin;ray.d=dv/length;ray.maxt=length-.0008
  clear=~scene.ray_test(ray);g=dr.select(clear,dr.rcp(dr.maximum(.0625,dr.square(length))),0)
  native=direct.faces(h,tr,tc).cpu().numpy()
  geom=g.numpy().reshape(130050,len(q)).mean(1)
  vis=clear.numpy().reshape(130050,len(q)).mean(1)
  pix=(geom[ix]*ww).sum(1).reshape(256,256)
  mine=(native[:,2][ix]*ww).sum(1).reshape(256,256)
  sel=(pix>0)&(mine>0);db=np.zeros_like(pix)
  db[sel]=10*np.log10(pix[sel]/mine[sel])
  result=dict(map=i,face_visibility_disagreement_fraction=float(np.mean(abs(vis-native[:,3])>1e-5)),
   face_visibility_abs_mean=float(np.mean(abs(vis-native[:,3]))),positive_pixel_log_difference_rms=float(np.sqrt(np.mean(db[sel]**2))),
   direct_power_relative_L1=float(np.sum(abs(geom-native[:,2]))/np.sum(geom)),
   pixel_zero_disagreement=int(np.sum((pix>0)!=(mine>0))))
  np.savez(O/f'visibility_{i}.npz',mitsuba_pixel=pix,native_pixel=mine,mi_face_visible=vis,cc_face_visible=native[:,3])
  records.append(result);print(json.dumps(result),flush=True)
 atomic_json(A/'visibility_crosscheck.json',dict(status='complete',records=records,GPU=gpu,solver_backend=mi.variant(),n=4,receiver_mesh_in_ray_query=False))
if __name__=='__main__':check()
