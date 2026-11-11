"""Analytic physics checks, then input-only CPU descriptor extraction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c19_fine_physics import transport,features,NAMES
from src.c12_material_features import slab
from src.c7_surface_features import tx_visibility
O=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
checks={}
h=np.zeros((256,256));p=np.array([128*256+156,64*256+64,128*256+100])
v=transport(h,128,100,p).reshape(-1,2,16)
assert (v[:,:,0]==0).all() and np.max(abs(v[:,:,6:14]))==0
checks['flat_surface_zero_hits_and_loss']=True
h=np.repeat(np.maximum(0,5-.5*abs(np.arange(256)-128))[None],256,0)
p=np.array([128*256+156]);v=transport(h,128,100,p).reshape(2,16)
assert (v[:,0]==2).all()
np.testing.assert_allclose(v[:,14],.4,atol=1e-12)
np.testing.assert_allclose(v[:,15],16/26,atol=1e-12)
k=np.array([56.,0.,-2.]);k/=np.linalg.norm(k)
ns=np.array([[-.5,0,1],[.5,0,1]]);ns/=np.linalg.norm(ns,axis=1)[:,None]
cos=abs(ns@k)
expected=[]
for b in [0,1]:
 te,tm,_,_=slab(cos,b)
 ev=float(-10*np.log10(tm).sum());eh=float(-10*np.log10(te).sum())
 np.testing.assert_allclose(v[:,6+4*b],ev,rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(v[:,7+4*b],eh,rtol=1e-10,atol=1e-10)
 expected.append({'V_dB':ev,'H_dB':eh})
rot=transport(np.rot90(h),255-100,128,np.array([(255-156)*256+128])).reshape(2,16)
np.testing.assert_allclose(rot[:,6:14],v[:,6:14],rtol=1e-10,atol=1e-10)
checks['analytic_two_facet_ridge']={'crossings':2,'first_t':.4,'last_t':16/26,'expected_losses':expected,'max_rotation_difference':float(np.max(abs(rot[:,6:14]-v[:,6:14])))}
for b in [0,1]:
 te,tm,re,rm=slab(np.linspace(.001,1,1000),b)
 assert np.max(te+re)<=1+1e-10 and np.max(tm+rm)<=1+1e-10
 np.testing.assert_allclose(slab(np.array([1.]),b)[0],slab(np.array([1.]),b)[1])
checks['slab_passivity_and_normal_incidence_polarization']=True
sel=json.loads((O/'selection.json').read_text());pixels=np.load(O/'pixels.npy')
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r');G=np.load(R/'features/c3/train.npy',mmap_mode='r')
j=0;row=rows[sel['index_rows'][j]];hh=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col']);tv=tx_visibility(np.asarray(G[sel['index_rows'][j]],np.float32),3-(hh[tr,tc]%1))[0]
test=features(hh,tr,tc,pixels[j,:8],tv)
checks['real_data_feature_shape']=list(test.shape);checks['feature_count']=len(NAMES)
(O/'implementation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'checks':'passed','features':len(NAMES)}),flush=True)
start=time.time()
out=np.lib.format.open_memmap(O/'physical_features.npy',mode='w+',dtype='float32',shape=(len(rows) if False else len(sel['index_rows']),256,5,len(NAMES)))
def one(j):
 i=sel['index_rows'][j];row=rows[i];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col'])
 rr,cc=np.divmod(pixels[j],256)
 pp=np.stack((pixels[j],(rr-1)*256+cc-1,(rr-1)*256+cc+1,(rr+1)*256+cc-1,(rr+1)*256+cc+1),1).ravel()
 tv=tx_visibility(np.asarray(G[i],np.float32),3-(h[tr,tc]%1))[0]
 x=features(h,tr,tc,pp,tv).reshape(256,5,len(NAMES));out[j]=x.astype('float32')
 return j
with ThreadPoolExecutor(max_workers=8) as pool:
 for n,j in enumerate(pool.map(one,range(len(sel['index_rows']))),1):
  if n%32==0:print(json.dumps({'maps':n,'total':len(sel['index_rows']),'seconds':time.time()-start}),flush=True)
out.flush()
source={}
for p in [R/'src/c19_fine_physics.py',R/'src/c19_triangle_transport.cpp',Path(__file__),R/'src/c7_surface_features.py']:
 source[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
meta={'state':'completed','shape':list(out.shape),'names':NAMES,'location_order':['query','NW1m','NE1m','SW1m','SE1m'],'CPU_workers':8,'seconds':time.time()-start,'source_sha256':source,'selection':'selection.json','no_labels_masks_or_predictions_in_features':True,'checks':'implementation_checks.json'}
(O/'physical_features.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
