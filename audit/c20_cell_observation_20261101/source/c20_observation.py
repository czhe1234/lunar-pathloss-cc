"""Candidate surface/measurement operators; all powers from input-only surrogate."""
from pathlib import Path
import ctypes
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1]
lib=ctypes.CDLL(str(R/'src/c20_triangle_transport.so'))
lib.triangle_transport_points.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
lib.triangle_transport_points.restype=None
RADII=np.array([2,4,6,8,12,16,20,24,32,40,48,64,80,96,112,128.])
EDGE=np.r_[0.,(RADII[:-1]+RADII[1:])/2,136.]
def quadrature(n):
 """Centroids of n*n congruent equal-area subtriangles, as barycentric weights."""
 uv=[]
 for i in range(n):
  for j in range(n-i):
   uv.append(((i+1/3)/n,(j+1/3)/n))
   if i+j<n-1:uv.append(((i+2/3)/n,(j+2/3)/n))
 uv=np.array(uv);w=np.c_[1-uv.sum(1),uv]
 assert len(w)==n*n and np.min(w)>0
 return w
def mesh(diag):
 rr,cc=np.mgrid[:255,:255];r=rr.ravel();c=cc.ravel()
 offsets=[[[0,0],[0,1],[1,1]],[[0,0],[1,0],[1,1]]] if diag==0 else [[[0,0],[0,1],[1,0]],[[0,1],[1,1],[1,0]]]
 verts=np.stack([np.stack((r[:,None]+np.array(off)[None,:,0],c[:,None]+np.array(off)[None,:,1]),-1) for off in offsets],1).reshape(-1,3,2)
 cent=verts.mean(1);tri=Delaunay(cent)
 return verts,cent,tri
def interpolation(tri,queries):
 simplex=tri.find_simplex(queries);assert (simplex>=0).all()
 transform=tri.transform[simplex];bary=np.einsum('nij,nj->ni',transform[:,:2],queries-transform[:,2]);w=np.c_[bary,1-bary.sum(1)]
 ids=tri.simplices[simplex].copy();w[abs(w)<1e-10]=0;assert w.min()>-1e-8;w=np.maximum(w,0);w/=w.sum(1)[:,None]
 for i in range(len(ids)):ids[i,w[i]==0]=ids[i,np.argmax(w[i])]
 return ids,w
def xyz_from_vertices(h,verts):
 return np.concatenate((verts,h[verts[:,:,0],verts[:,:,1],None]+1),axis=2).astype(float)
def surface(h,r,c,diag):
 r=np.clip(r,0,255-1e-8);c=np.clip(c,0,255-1e-8);iy=r.astype(int);ix=c.astype(int);v=r-iy;u=c-ix
 a=h[iy,ix];b=h[iy,ix+1];z=h[iy+1,ix];d=h[iy+1,ix+1]
 if diag==0:
  side=v<=u;sx=np.where(side,b-a,d-z);sy=np.where(side,d-b,z-a);anchor=a
 else:
  side=u+v<=1;sx=np.where(side,b-a,d-z);sy=np.where(side,z-a,d-b);anchor=np.where(side,a,b+z-d)
 return anchor+sx*u+sy*v,sx,sy
def direct(h,tr,tc,xyz,diag):
 h=np.ascontiguousarray(h,np.float64);xyz=np.ascontiguousarray(xyz,np.float64);out=np.empty((len(xyz),16),np.float64)
 lib.triangle_transport_points(h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),int(tr),int(tc),xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(xyz),diag,out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
 assert np.isfinite(out).all()
 return out
def field(h,tr,tc,xyz,diag,txvis,naz=32):
 """Return absolute direct/diffuse path gains for both bands, and LOS flag."""
 n=len(xyz);rr,cc,zrx=xyz.T;zt=np.floor(h[tr,tc])+3
 ang=2*np.pi*np.arange(naz)/naz;ox=np.cos(ang)[:,None]*RADII;oy=np.sin(ang)[:,None]*RADII
 qx=cc[:,None,None]+ox;qy=rr[:,None,None]+oy
 inside=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255)
 hh,sx,sy=surface(h,qy,qx,diag);norm=np.sqrt(1+sx*sx+sy*sy)
 rz=zrx[:,None,None]-(hh+.05);rlen=np.sqrt(RADII**2+rz*rz)
 coso=np.maximum((sx*ox+sy*oy+rz)/norm/rlen,0)
 slope=(hh-zrx[:,None,None])/RADII
 prior=np.concatenate((np.full((n,naz,1),-1e6),np.maximum.accumulate(slope,2)[:,:,:-1]),2)
 rv=(slope+.05/RADII>=prior)*inside
 tv=map_coordinates(txvis,np.stack((qy,qx)),order=1,mode='constant',cval=0,prefilter=False)
 tz=zt-(hh+.05);txx=tc-qx;txy=tr-qy;tlen=np.sqrt(txx*txx+txy*txy+tz*tz).clip(.25)
 cosi=np.maximum((-sx*txx-sy*txy+tz)/norm/tlen,0)
 area=np.pi*(EDGE[1:]**2-EDGE[:-1]**2)/naz
 w=area*norm*cosi*coso/(tlen*tlen*rlen*rlen)/np.pi*rv*tv
 out=direct(h,tr,tc,xyz,diag)
 d2=np.maximum((rr-tr)**2+(cc-tc)**2+(zrx-zt)**2,.25**2)
 results=[]
 for b,(f,sig) in enumerate([(415e6,.0003562),(5.775e9,.009324)]):
  lam2=(299792458./f/(4*np.pi))**2
  T=np.power(10.,-np.minimum(out[:,6+4*b],3000)/10)/d2*lam2
  eta=3.411-1j*sig/(2*np.pi*f*8.8541878128e-12);root=np.sqrt(eta-(1-cosi*cosi))
  rs=(cosi-root)/(cosi+root);rp=(eta*cosi-root)/(eta*cosi+root)
  # Fixed one-bounce approximation uses semi-infinite reflection, as C19.
  D=(w*(abs(rs)**2+abs(rp)**2)/2).sum((1,2))*.632**2*lam2
  results.extend((T,D))
 results.append(out[:,0]==0)
 a=np.column_stack(results);assert np.isfinite(a).all() and a.min()>=0
 return a
def evaluate_map(h,tr,tc,query_rc,verts,tri,txvis,diag,orders=(2,4),naz=32,precomputed=None):
 ids,weights=interpolation(tri,query_rc) if precomputed is None else precomputed
 unique,inverse=np.unique(ids,return_inverse=True);idx=inverse.reshape(ids.shape)
 vv=xyz_from_vertices(h,verts[unique]);cent=vv.mean(1)
 qh,_,_=surface(h,query_rc[:,0],query_rc[:,1],diag)
 qxyz=np.c_[query_rc,qh+1];result={'point':field(h,tr,tc,qxyz,diag,txvis,naz)}
 # Restrict vector size to keep eight CPU workers below a fewGB.
 def batch(x):
  return np.concatenate([field(h,tr,tc,x[i:i+1024],diag,txvis,naz) for i in range(0,len(x),1024)])
 def interp(values):return (values[idx]*weights[:,:,None]).sum(1)
 raw={'centroid':batch(cent)}
 for order in orders:
  xyz=np.einsum('qv,tvc->tqc',quadrature(order),vv).reshape(-1,3)
  raw[f'area{order*order}']=batch(xyz).reshape(len(unique),order*order,5).mean(1)
 for name,value in raw.items():
  result[name]=interp(value)
  power=np.stack((value[:,0]+value[:,1],value[:,2]+value[:,3]),1)
  result[name+'_dbinterp']=(-10*np.log10(power.clip(1e-30))[idx]*weights[:,:,None]).sum(1)
  # Constant received diffuse power at each query isolates direct-cell effects.
  ds=result[name].copy();ds[:,1]=result['point'][:,1];ds[:,3]=result['point'][:,3]
  result[name+'_fixed_diffuse']=ds
 return result
