"""Published-input physics hypotheses. No labels, masks, propagation engine, or actual-path claim."""
from pathlib import Path
import ctypes
import numpy as np
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parents[1]
lib=ctypes.CDLL(str(R/'src/c19_triangle_transport.so'))
lib.triangle_transport.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
lib.triangle_transport.restype=None
TN=['crossings','minimum_cos','sum_log_inverse_cos','incidence_plane_turn','minimum_Brewster_distance','minimum_crossing_separation','V415','H415','reset415','fixed415','V58','H58','reset58','fixed58','first_t','last_t']
TRANSPORT_NAMES=[f'triangle{d}_{n}' for d in [0,1] for n in TN]
RADII=np.array([2,4,6,8,12,16,20,24,32,40,48,64,80,96,112,128.])
EDGE=np.r_[0.,(RADII[:-1]+RADII[1:])/2,136.];AREA=np.pi*(EDGE[1:]**2-EDGE[:-1]**2)/16
PROJECTION_NAMES=['triangle_area_log_mean','triangle_area_log_std','normal_spread','direct_log_J_mean','direct_log_J_std','direct_log_J_max','direct_grazing_fraction','direct_signed_cos','surface_log_weight','surface_log_inverse_cos_mean','surface_grazing01','surface_grazing03','surface_grazing10','arrival_entropy','arrival_effective_count','arrival_cos_mean','arrival_strongest_cos','diffuse415','diffuse58']+[f'azimuth_weight_{j:02d}' for j in range(16)]
NAMES=TRANSPORT_NAMES+PROJECTION_NAMES
def transport(h,tr,tc,pixels):
 h=np.ascontiguousarray(h,np.float64);p=np.ascontiguousarray(pixels,np.int32);out=np.empty((len(p),32),np.float64)
 lib.triangle_transport(h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),int(tr),int(tc),p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),len(p),out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
 assert np.isfinite(out).all()
 return out
def features(h,tr,tc,pixels,txvis):
 h=np.asarray(h,np.float64);rr,cc=np.divmod(pixels,256);n=len(rr);sy,sx=np.gradient(h)
 zrx=h[rr,cc]+1;zt=np.floor(h[tr,tc])+3.;v=np.stack((cc-tc,rr-tr,zrx-zt),1);dist=np.linalg.norm(v,axis=1).clip(.25);k=v/dist[:,None]
 offsets=np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]])
 vec=[]
 for dr,dc in offsets:vec.append(np.stack((np.full(n,dc),np.full(n,dr),h[np.clip(rr+dr,0,255),np.clip(cc+dc,0,255)]-h[rr,cc]),1))
 vec=np.stack(vec,1);cross=np.cross(vec,np.roll(vec,-1,axis=1));norm=np.linalg.norm(cross,axis=2);normals=cross/norm[:,:,None];areas=norm/2
 cos=np.abs((normals*k[:,None]).sum(2));J=1/(areas*np.maximum(cos,1e-4));avg=normals.mean(1)
 nc=np.stack((-sx[rr,cc],-sy[rr,cc],np.ones(n)),1);nc/=np.linalg.norm(nc,axis=1)[:,None]
 vals=[np.log(areas).mean(1),np.log(areas).std(1),((normals-avg[:,None])**2).sum(2).mean(1),np.log(J).mean(1),np.log(J).std(1),np.log(J).max(1),(cos<.03).mean(1),(nc*k).sum(1)]
 ang=2*np.pi*np.arange(16)/16;ox=np.cos(ang)[:,None]*RADII;oy=np.sin(ang)[:,None]*RADII
 qx=cc[:,None,None]+ox;qy=rr[:,None,None]+oy;inside=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255);coords=np.stack((qy,qx))
 hh,ssx,ssy=[map_coordinates(f,coords,order=1,mode='nearest',prefilter=False) for f in [h,sx,sy]]
 normalnorm=np.sqrt(1+ssx**2+ssy**2);rz=zrx[:,None,None]-(hh+.05);rlen=np.sqrt(RADII[None,None]**2+rz**2)
 coso=np.maximum((ssx*ox+ssy*oy+rz)/normalnorm/rlen,0)
 slope=(hh-zrx[:,None,None])/RADII;prior=np.concatenate((np.full((n,16,1),-1e6),np.maximum.accumulate(slope,2)[:,:,:-1]),2);rxvis=(slope+.05/RADII>=prior)*inside
 tv=map_coordinates(txvis,coords,order=1,mode='constant',cval=0,prefilter=False)
 txz=zt-(hh+.05);txx=tc-qx;txy=tr-qy;tlen=np.sqrt(txx**2+txy**2+txz**2).clip(.25)
 cosi=np.maximum((-ssx*txx-ssy*txy+txz)/normalnorm/tlen,0)
 w=AREA*normalnorm*cosi*coso/(tlen**2*rlen**2)*(dist**2)[:,None,None]/np.pi*rxvis*tv
 total=w.sum((1,2));den=total.clip(1e-30);cosrx=np.abs((-ox*nc[:,None,None,0]-oy*nc[:,None,None,1]+rz*nc[:,None,None,2])/rlen)
 mean=lambda a:(w*a).sum((1,2))/den
 az=w.sum(2)/den[:,None];best=w.reshape(n,-1).argmax(1);bestcos=np.take_along_axis(cosrx.reshape(n,-1),best[:,None],axis=1)[:,0]
 diffuse=[]
 for f,sig in [(415e6,.0003562),(5.775e9,.009324)]:
  eta=3.411-1j*sig/(2*np.pi*f*8.8541878128e-12);root=np.sqrt(eta-(1-cosi*cosi));rs=(cosi-root)/(cosi+root);rp=(eta*cosi-root)/(eta*cosi+root);refl=(abs(rs)**2+abs(rp)**2)/2
  diffuse.append((w*refl).sum((1,2))*.632**2)
 vals.extend([np.log(total+1e-20),mean(-np.log(np.maximum(cosrx,1e-4))),mean(cosrx<.01),mean(cosrx<.03),mean(cosrx<.1),
 -(az*np.log(az+1e-30)).sum(1),total**2/(w*w).sum((1,2)).clip(1e-30),mean(cosrx),np.where(total>0,bestcos,0),*diffuse])
 vals.extend([az[:,j] for j in range(16)])
 out=np.column_stack((transport(h,tr,tc,pixels),*vals))
 assert out.shape==(n,len(NAMES)) and np.isfinite(out).all()
 return out
