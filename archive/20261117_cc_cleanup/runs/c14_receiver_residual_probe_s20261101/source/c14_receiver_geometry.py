"""Input-only receiver neighborhood and subcell line-profile descriptors.
The five receiver positions are geometric probes, not recovered simulation cells.
"""
import numpy as np
from scipy.ndimage import map_coordinates
PATCH=(-4,-2,-1,0,1,2,4)
CELL=((0.,0.),(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))
BASE=['margin','critical_RX','critical_TX','buried_length','positive_fraction','max_location','crossings','incidence_cos','endpoint_height_delta']
NAMES=[f'RX_patch_{r}_{c}' for r in PATCH for c in PATCH]+[f'RX_cell{k}_{n}' for k in range(len(CELL)) for n in BASE]+[f'RX_cell_visible_fraction_margin_{z}' for z in [-.2,-.05,0,.05,.2]]+['RX_cell_margin_mean','RX_cell_margin_std','RX_cell_margin_min','RX_cell_margin_max']
def fields(h):
 sy,sx=np.gradient(h)
 return h,sx,sy
def features(f,tr,tc,pixels,steps=256):
 h,sx,sy=f;rr,cc=np.divmod(pixels,256);n=len(pixels)
 ht=float(h[tr,tc]);txz=np.floor(ht)+3.;hr=h[rr,cc].astype(float)
 result=[]
 for dr in PATCH:
  for dc in PATCH:
   z=h[np.clip(rr+dr,0,255),np.clip(cc+dc,0,255)].astype(float)-hr
   result.append(np.arcsinh(z/2)/3)
 t=(np.arange(steps,dtype=np.float64)+.5)/steps;ts=np.r_[0.,t,1.];dt=np.diff(ts);margins=[]
 for dr,dc in CELL:
  ry=np.clip(rr+dr,0,255);rx=np.clip(cc+dc,0,255);rxh=map_coordinates(h,np.stack((ry,rx)),order=1,mode='nearest',prefilter=False).astype(float);rz=rxh+1
  dx=rx-tc;dy=ry-tr;dz=rz-txz;distance=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
  xx=tc+dx[:,None]*t;yy=tr+dy[:,None]*t
  hh=map_coordinates(h,np.stack((yy,xx)),order=1,mode='nearest',prefilter=False)
  e=hh-(txz+dz[:,None]*t);margin=e.max(1);margins.append(margin)
  ee=np.concatenate((np.full((n,1),ht-txz),e,np.full((n,1),-1.)),1);a,z=ee[:,:-1],ee[:,1:]
  length=((np.maximum(a,0)+np.maximum(z,0))/np.maximum(abs(a)+abs(z),1e-9)*dt).sum(1)*distance
  gx=map_coordinates(sx,np.stack((ry,rx)),order=1,mode='nearest',prefilter=False);gy=map_coordinates(sy,np.stack((ry,rx)),order=1,mode='nearest',prefilter=False)
  cos=(dz-gx*dx-gy*dy)/distance/np.sqrt(1+gx*gx+gy*gy)
  vals=[np.arcsinh(margin/2)/3,np.arcsinh((e/t).max(1)/2)/3,np.arcsinh((e/(1-t)).max(1)/2)/3,np.log1p(length)/6,(e>0).mean(1),t[e.argmax(1)],((a>0)!=(z>0)).sum(1),cos,np.arcsinh((rxh-hr)/2)/3]
  result.extend(vals)
 margins=np.stack(margins,1)
 result.extend([(margins<z).mean(1) for z in [-.2,-.05,0,.05,.2]])
 result.extend([np.arcsinh(z/2)/3 for z in [margins.mean(1),margins.std(1),margins.min(1),margins.max(1)]])
 out=np.stack(result,1).astype(np.float32);assert out.shape==(n,len(NAMES)) and np.isfinite(out).all()
 return out
