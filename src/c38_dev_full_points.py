"""C38 full-TRAIN refit preparation: expand internal DEV to1024points/map. retain original256 and add768 uniform pixels/map.
All features are deterministic C10 H/TX descriptors; no masks/targets select pixels.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates,uniform_filter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c8_tx_context import terrain_fields,tx_features
from src.c7_surface_features import features as surface,terrain_fields as surface_fields,tx_visibility
from src.c10_phase_geometry import phase_features
O=R/'features/c38_dev_full';O.mkdir(exist_ok=False)
D=R.parent/'data';P=R/'features/c6_points'
rows=list(csv.DictReader((D/'train_index.csv').open()))
H=np.load(D/'train_hm.npy',mmap_mode='r')
G=[np.load(R/f'features/{n}/train.npy',mmap_mode='r') for n in ['c3','c4_extra']]
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
ids=np.load(P/'dev_map_indices.npy');oldpix=np.load(P/'dev_pixel.npy').reshape(len(ids),256)
oldx=np.load(R/'features/c10_points/dev_x.npy',mmap_mode='r')
@lru_cache(maxsize=128)
def fields(hi):
 h=np.array(H[hi],np.float32)
 return h,terrain_fields(h),h-uniform_filter(h,5,mode='nearest'),surface_fields(h)
def features(i,pix):
 r=rows[int(i)];h,tf,hp,sf=fields(int(r['hm_row']))
 tr=int(r['tx_row']);tc=int(r['tx_col']);ht=float(h[tr,tc]);rr,cc=np.divmod(pix,256)
 dx=cc.astype(np.float32)-tc;dy=rr.astype(np.float32)-tr;dz=h[rr,cc]+1-ht-3
 dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
 basic=np.stack([h[rr,cc]/100,(h[rr,cc]-ht)/50,hp[rr,cc]/5,tf[1][rr,cc]/5,tf[2][rr,cc]/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((rr==tr)&(cc==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)],1)
 gg=np.concatenate([np.asarray(a[i,:,rr,cc],np.float32) for a in G],1)
 ctx=np.broadcast_to(tx_features(tf,tr,tc),(len(pix),73))
 t=(np.arange(128,dtype=np.float32)+.5)/128
 hh=map_coordinates(h,np.stack((tr+dy[:,None]*t,tc+dx[:,None]*t)),order=1,mode='nearest',prefilter=False)
 profile=np.arcsinh((hh-(ht+3+dz[:,None]*t))/2)/3
 vis=[tx_visibility(G[0][i],v) for v in [3,2]]
 surf=surface(h,sf,vis,tr,tc,pix)
 v=np.concatenate((basic,gg,ctx),1)
 critical=2.5+2*np.sinh(3*v[:,27]);phases=np.arange(8)/8
 phase8=np.arcsinh((critical[:,None]-(3-np.remainder(v[:,31,None]+phases,1)))/2)/3
 v=np.concatenate((v,profile,surf,phase8),1).astype(np.float32)
 x=np.concatenate((v,phase_features(v)),1)
 fs=(20*np.log10(4*np.pi*dist[:,None]*np.array([415e6,5.775e9])[None]/299792458.)).astype(np.float32)
 return x,fs
checks=[]
for j in [0,100,1000,len(ids)-1]:
 a,_=features(int(ids[j]),oldpix[j]);d=abs(a-oldx[j*256:(j+1)*256])
 checks.append({'dev_map_position':j,'max_abs_difference':float(d.max())})
assert max(c['max_abs_difference'] for c in checks)<1e-4,checks
(O/'replay.json').write_text(json.dumps(checks,indent=2)+'\n')
n=1024;N=len(ids)*n
spec={'x':(np.float32,(N,294)),'y':(np.float32,(N,2)),'mask':(np.uint8,(N,2)),'fspl':(np.float32,(N,2)),'pixel':(np.int32,(N,))}
out={k:np.lib.format.open_memmap(O/f'dev_{k}.npy',mode='w+',dtype=t,shape=shape) for k,(t,shape) in spec.items()}
old={k:np.load(P/f'dev_{k}.npy',mmap_mode='r') for k in ['y','mask','fspl']}
def item(j):
 i=int(ids[j]);rng=np.random.default_rng(20261207+i*104729)
 keep=np.ones(65536,bool);keep[oldpix[j]]=False
 new=rng.choice(np.flatnonzero(keep),768,replace=False).astype(np.int32)
 x,fs=features(i,new);rr,cc=np.divmod(new,256)
 y=np.stack([np.asarray(v[i,rr,cc],np.float32) for v in Y],1)
 m=np.stack([np.unpackbits(v[i],bitorder='big').reshape(256,256)[rr,cc] for v in M],1)
 return j,new,x,fs,y,m
t0=time.time()
with ThreadPoolExecutor(max_workers=12) as pool:
 for lo in range(0,len(ids),192):
  for j,pix,x,fs,y,m in pool.map(item,range(lo,min(lo+192,len(ids)))):
   first=slice(j*1024,j*1024+256);last=slice(j*1024+256,(j+1)*1024);prev=slice(j*256,(j+1)*256)
   out['x'][first]=oldx[prev];out['x'][last]=x
   for k,v in [('y',y),('mask',m),('fspl',fs)]:
    out[k][first]=old[k][prev];out[k][last]=v
   out['pixel'][first]=oldpix[j];out['pixel'][last]=pix
  if lo%1920==0:
   p={'maps_done':min(lo+192,len(ids)),'maps_total':len(ids),'seconds':time.time()-t0}
   (O/'progress.json').write_text(json.dumps(p)+'\n');print(json.dumps(p),flush=True)
for v in out.values():v.flush()
np.save(O/'dev_map_indices.npy',ids)
meta={'complete':True,'dev_maps':len(ids),'points_per_map':1024,'dev_points':N,'original256_retained_bitwise':True,'additional768':'uniform without replacement excluding original256; no target/mask selection','features':'Same294 C10 descriptors','use':'Only full-TRAIN refit after complete DEV promotion; not used in FIT training or selection','valid_points':out['mask'].sum(0).tolist(),'seconds':time.time()-t0}
(O/'complete.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
