"""CPU FIT-only triage: exact ray clearance on two possible regular triangulations.
No original mesh is asserted recovered; no training or ray-power simulation.
"""
from pathlib import Path
import json,csv,time,os,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';rows=list(csv.DictReader((D/'train_index.csv').open()));dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids']);seen=set();unique=[]
for i,r in enumerate(rows):
 if r['terrain_id'] not in dev and r['terrain_id'] not in seen:unique.append(i);seen.add(r['terrain_id'])
ids=np.array(unique)[np.linspace(0,len(unique)-1,64,dtype=int)]
hm=np.load(D/'train_hm.npy',mmap_mode='r');geo=np.load(R/'features/c3/train.npy',mmap_mode='r');rr,cc=np.indices((32,32));rr=(rr.ravel()*8+4);cc=(cc.ravel()*8+4)

def triangle(h,y,x,diag):
 x0=np.floor(x).astype(int).clip(0,254);y0=np.floor(y).astype(int).clip(0,254);fx=x-x0;fy=y-y0
 a=h[y0,x0];b=h[y0,x0+1];c=h[y0+1,x0];d=h[y0+1,x0+1]
 if diag=='sum':return np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,d+(c-d)*(1-fx)+(b-d)*(1-fy))
 return np.where(fy<=fx,a+(b-a)*fx+(d-b)*fy,a+(c-a)*fy+(d-c)*fx)

def exact(h,tr,tc,diag):
 dx=cc-tc;dy=rr-tr;z0=h[tr,tc]+3;z1=h[rr,cc]+1;best=np.full(len(rr),-1.,dtype=float)
 for den,steps in [(np.abs(dx),255),(np.abs(dy),255),(np.abs(dx+dy if diag=='sum' else dx-dy),510)]:
  t=np.arange(1,steps+1)[None]/np.maximum(den[:,None],1)
  valid=(den[:,None]>0)&(t<1);t=np.where(valid,t,0.)
  x=tc+dx[:,None]*t;y=tr+dy[:,None]*t
  e=triangle(h,y,x,diag)-(z0+(z1-z0)[:,None]*t);e=np.where(valid,e,-np.inf)
  best=np.maximum(best,e.max(1))
 return best

def one(i):
 row=rows[i];h=hm[int(row['hm_row'])].astype(float);tr=int(row['tx_row']);tc=int(row['tx_col'])
 old=(2*np.sinh(3*geo[i,:8,rr,cc].astype(float))).max(-1)
 # NumPy mixed advanced indexing returns [pixels, channels].
 assert old.shape==rr.shape
 a=exact(h,tr,tc,'sum');b=exact(h,tr,tc,'diff')
 return {'index_row':int(i),'terrain_id':row['terrain_id']},np.stack((old,a,b))
start=time.time()
# Analytic plane check: all three surfaces coincide and remain clear.
h=np.indices((256,256))[0]*.25+np.indices((256,256))[1]*.125
for diag in ['sum','diff']:assert np.allclose(exact(h,128,128,diag),-1.,atol=1e-10)
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,ids))
a=np.stack([z[1] for z in results]);summary={}
for j,name in [(1,'sum_diagonal'),(2,'difference_diagonal')]:
 e=a[:,j]-a[:,0];flip=(a[:,j]>=0)!=(a[:,0]>=0)
 summary[name]={'clearance_abs_difference_m':{str(q):float(np.quantile(np.abs(e),q)) for q in [.5,.9,.99,1.]},'visibility_sign_changed_pixels':int(flip.sum()),'sampled_pixels':int(flip.size),'visibility_sign_change_fraction':float(flip.mean())}
summary['between_triangulations']={'sign_change_fraction':float(((a[:,1]>=0)!=(a[:,2]>=0)).mean()),'max_abs_clearance_m':float(np.max(np.abs(a[:,1]-a[:,2])))}
O=R/'audit/c5_exact_mesh_triage';O.mkdir(exist_ok=True);np.savez_compressed(O/'sampled_clearances.npz',clearance=a,rr=rr,cc=cc,index_rows=ids)
res={'scope':'64 distinct FIT terrains, 1024 fixed pixels per map, input-only geometric diagnostic; no official validation and no fitting','exactness':'For each hypothetical piecewise-planar mesh, maxima at grid-axis/diagonal breakpoints plus endpoints. Does not establish actual organizer mesh.','selection':[z[0] for z in results],'summary':summary,'plane_check_passed':True,'seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(O/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='selection'}))
