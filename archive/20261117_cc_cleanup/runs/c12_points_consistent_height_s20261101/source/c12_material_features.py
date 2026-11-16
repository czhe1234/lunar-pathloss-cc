"""C12 adapts the existing independent CC C7 descriptor to one supplied TX height."""
"""Input-only surface-intersection descriptors and fixed material responses.
This is a feature extractor, not a propagation solver or label generator.
The published raster is interpolated; original triangles are unavailable.
"""
import numpy as np
from scipy.ndimage import map_coordinates
FREQ=(415e6,5.775e9);SIGMA=(.0003562,.009324);EPS=8.8541878128e-12
def slab(cosine,band,thickness=10.,sigma=None):
 c=np.clip(np.abs(cosine),1e-5,1).astype(np.float64)
 eta=3.411-1j*(SIGMA[band] if sigma is None else sigma)/(2*np.pi*FREQ[band]*EPS)
 a=np.sqrt(eta-(1-c*c))
 rp=(c-a)/(c+a);rm=(eta*c-a)/(eta*c+a)
 q=2*np.pi*thickness/(299792458./FREQ[band])*a
 e=np.exp(-1j*q);e2=e*e
 t0=(1-rp*rp)*e/(1-rp*rp*e2);t1=(1-rm*rm)*e/(1-rm*rm*e2)
 r0=rp*(1-e2)/(1-rp*rp*e2);r1=rm*(1-e2)/(1-rm*rm*e2)
 return abs(t0)**2,abs(t1)**2,abs(r0)**2,abs(r1)**2
BASE=['crossings','first_t','last_t','minimum_cos','mean_cos','mean_entry_cos','mean_exit_cos','sum_log_cos','maximum_incident_slope','crossing_t_std']
NAMES=[f'tx{ht}_{v}' for ht in [3,2] for v in BASE+['slab_TE_415','slab_TM_415','slab_TE_58','slab_TM_58','slab_mix_415','slab_mix_58']]
def features(h,tr,tc,pixels,steps=384,txheight=3.):
 rr,cc=np.divmod(pixels,256);dx=cc.astype(float)-tc;dy=rr.astype(float)-tr
 t=np.linspace(0,1,steps+1);qx=tc+dx[:,None]*t;qy=tr+dy[:,None]*t
 hh=map_coordinates(h,np.stack((qy,qx)),order=1,mode='nearest',prefilter=False)
 result=[]
 for ht in [txheight]:
  dz=h[rr,cc]+1-float(h[tr,tc])-ht
  e=hh-(float(h[tr,tc])+ht+dz[:,None]*t)
  cross=(e[:,:-1]>0)!=(e[:,1:]>0)
  p,k=np.nonzero(cross);n=len(pixels);num=np.bincount(p,minlength=n).astype(float)
  den=e[p,k+1]-e[p,k];frac=-e[p,k]/np.where(abs(den)>1e-20,den,1)
  tt=(k+np.clip(frac,0,1))/steps
  xx=np.clip(tc+dx[p]*tt,0,255-1e-7);yy=np.clip(tr+dy[p]*tt,0,255-1e-7)
  ix=xx.astype(int);iy=yy.astype(int);u=xx-ix;v=yy-iy
  # Derivative of the same bilinear surface used for locating intersections.
  sx=(h[iy,ix+1]-h[iy,ix])*(1-v)+(h[iy+1,ix+1]-h[iy+1,ix])*v
  sy=(h[iy+1,ix]-h[iy,ix])*(1-u)+(h[iy+1,ix+1]-h[iy,ix+1])*u
  norm=np.sqrt(1+sx*sx+sy*sy);dist=np.sqrt(dx[p]**2+dy[p]**2+dz[p]**2).clip(.25)
  cos=np.abs(dz[p]-sx*dx[p]-sy*dy[p])/norm/dist;cos=np.clip(cos,1e-5,1)
  entry=e[p,k+1]>0
  first=np.ones(n);last=np.zeros(n);mn=np.ones(n);ms=np.zeros(n)
  np.minimum.at(first,p,tt);np.maximum.at(last,p,tt);np.minimum.at(mn,p,cos);np.maximum.at(ms,p,np.sqrt(sx*sx+sy*sy))
  count=num.clip(1);summ=lambda a:np.bincount(p,weights=a,minlength=n)
  en=np.bincount(p[entry],minlength=n).clip(1);ex=np.bincount(p[~entry],minlength=n).clip(1)
  vals=[num,first,last,mn,summ(cos)/count,np.bincount(p[entry],weights=cos[entry],minlength=n)/en,np.bincount(p[~entry],weights=cos[~entry],minlength=n)/ex,summ(np.log(cos)),ms,np.sqrt(np.maximum(summ(tt**2)/count-(summ(tt)/count)**2,0))]
  trans=[]
  for b in [0,1]:
   te,tm,_,_=slab(cos,b)
   trans += [summ(-10*np.log10(te.clip(1e-30))),summ(-10*np.log10(tm.clip(1e-30)))]
  vals += trans
  for b in [0,1]:
   a,z=trans[2*b:2*b+2];low=np.minimum(a,z)
   vals.append(low-10*np.log10((10**(-(a-low)/10)+10**(-(z-low)/10))/2))
  result.extend(vals)
 out=np.stack(result,1).astype(np.float32)
 assert out.shape==(len(pixels),len(BASE)+6) and np.isfinite(out).all()
 return out

