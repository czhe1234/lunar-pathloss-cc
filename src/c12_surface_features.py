"""C12 adapts the existing independent CC C7 descriptor to one supplied TX height."""
"""Input-only off-axis surface aggregation. Geometry proxies, no simulated labels.
No propagation engine, multi-bounce path tracing, target or mask is used.
"""
import numpy as np
from scipy.ndimage import map_coordinates
RADII=np.array([2,4,6,8,12,16,20,24,32,40,48,64,80,96,112,128.])
RAD_EDGES=np.r_[0.,(RADII[:-1]+RADII[1:])/2,136.]
AREA=np.pi*(RAD_EDGES[1:]**2-RAD_EDGES[:-1]**2)/16
BASE=['geom_weight','geom_near','geom_mid','geom_far','geom_lower_visibility','geom_upper_visibility','geom_max','excess_weighted','range_weighted','TX_range_weighted','incidence_weighted','exit_angle_weighted','alignment_weighted','alignment_sharp_weight','azimuth_cos_weighted','azimuth_sin_weighted','visible_fraction','visible_area','strongest_excess','strongest_radius']
NAMES=[f'surface_TX{ht}_{name}' for ht in [3,2] for name in BASE]
def terrain_fields(h):
 sy,sx=np.gradient(h)
 return np.stack((h,sx,sy))
def tx_visibility(g,txh):
 # Each C3 bin stores the maximum clearance, not its location. Midpoint
 # conversion to a ground+.05 endpoint has <=0.060m uncertainty for TX3.
 b=np.sinh(g[:8].astype(np.float32)*3)*2
 t=(np.arange(8,dtype=np.float32)+.5)/8
 correction=.95*t+(3-txh)*(1-t)
 em=np.max(b+correction[:,None,None],axis=0)
 tol=abs(.95-(3-txh))/16
 sig=lambda z:1/(1+np.exp(np.clip(z/.03,-60,60)))
 return np.stack((sig(em),sig(em+tol),sig(em-tol)))
def features(h,fields,vis,tr,tc,pixels,txheight=3.):
 rr,cc=np.divmod(pixels,256);n=len(pixels)
 ang=2*np.pi*np.arange(16)/16
 ox=np.cos(ang)[:,None]*RADII;oy=np.sin(ang)[:,None]*RADII
 qx=cc[:,None,None]+ox;qy=rr[:,None,None]+oy
 inside=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255)
 coords=np.stack((qy,qx))
 hh,sx,sy=[map_coordinates(f,coords,order=1,mode='nearest',prefilter=False) for f in fields]
 norm=np.sqrt(1+sx*sx+sy*sy)
 zrx=h[rr,cc]+1
 rz=zrx[:,None,None]-(hh+.05);rlen=np.sqrt(RADII[None,None,:]**2+rz**2)
 coso=np.maximum((sx*ox[None]+sy*oy[None]+rz)/norm/rlen,0)
 slope=(hh-zrx[:,None,None])/RADII
 prior=np.concatenate((np.full((n,16,1),-1e6),np.maximum.accumulate(slope,2)[:,:,:-1]),2)
 rxvis=(slope+.05/RADII>=prior)*inside
 result=[]
 for ti,ht in enumerate([txheight]):
  vv=[map_coordinates(v,coords,order=1,mode='constant',cval=0,prefilter=False) for v in vis[ti]]
  txz=float(h[tr,tc])+ht-(hh+.05);txx=tc-qx;txy=tr-qy
  tlen=np.sqrt(txx*txx+txy*txy+txz*txz).clip(.25)
  cosi=np.maximum((-sx*txx-sy*txy+txz)/norm/tlen,0)
  d0sq=(cc-tc)**2+(rr-tr)**2+(zrx-float(h[tr,tc])-ht)**2
  geom=AREA[None,None,:]*norm*cosi*coso/(tlen**2*rlen**2)*d0sq[:,None,None]/np.pi
  w=geom*rxvis*vv[0];wl=geom*rxvis*vv[1];wu=geom*rxvis*vv[2]
  extra=np.maximum(tlen+rlen-np.sqrt(d0sq)[:,None,None],0)
  # Dot between reflected TX->patch direction and patch->RX direction.
  dot_k=(-txx*(-ox[None])-txy*(-oy[None])-txz*rz)/(tlen*rlen)
  align=np.clip(dot_k+2*cosi*coso,-1,1)
  sm=lambda a:a.sum((1,2))
  total=sm(w);den=total.clip(1e-20)
  mean=lambda a:sm(w*a)/den
  best=w.reshape(n,-1).argmax(1)
  mx=w.reshape(n,-1).max(1)
  bx=np.take_along_axis(extra.reshape(n,-1),best[:,None],1)[:,0]
  br=RADII[best%16]
  log=lambda a:np.log10(a+1e-16)
  vals=[log(total),log(sm(w[:,:,:5])),log(sm(w[:,:,5:10])),log(sm(w[:,:,10:])),log(sm(wl)),log(sm(wu)),log(mx),
    np.log1p(mean(extra)),np.log1p(mean(np.broadcast_to(RADII,w.shape))),np.log1p(mean(tlen)),mean(cosi),mean(coso),mean(align),log(sm(w*np.maximum(align,0)**32)),
    mean(np.cos(ang)[None,:,None]),mean(np.sin(ang)[None,:,None]),sm(rxvis*vv[0])/np.maximum(inside.sum((1,2)),1),np.log1p(sm(rxvis*vv[0]*AREA[None,None,:]*norm)),np.where(total>0,np.log1p(bx),0),np.where(total>0,np.log1p(br),0)]
  result.extend(vals)
 out=np.stack(result,1).astype(np.float32)
 assert out.shape==(n,len(BASE)) and np.isfinite(out).all()
 return out

