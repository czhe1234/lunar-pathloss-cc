"""GPU implementation of C6 ordered profile and C7 surface descriptors.
Preserves the CPU feature definitions; raw input and geometry only.
"""
import numpy as np
import torch
from torch.nn import functional as F
from src.c7_surface_features import RADII,AREA
@torch.inference_mode()
def profile_gpu(h,tr,tc):
 yy,xx=torch.meshgrid(torch.arange(256,device=h.device,dtype=torch.float32),torch.arange(256,device=h.device,dtype=torch.float32),indexing='ij')
 dx=(xx-tc).flatten();dy=(yy-tr).flatten();dz=(h+1-float(h[tr,tc])-3).flatten()
 t=(torch.arange(128,device=h.device,dtype=torch.float32)+.5)/128
 out=[]
 hd=h.double()[None,None]
 for lo in range(0,65536,8192):
  x=tc+dx[lo:lo+8192,None]*t;y=tr+dy[lo:lo+8192,None]*t
  grid=torch.stack((x.double()/127.5-1,y.double()/127.5-1),-1)[None]
  hh=F.grid_sample(hd,grid,align_corners=True,padding_mode='border')[0,0].float()
  e=hh-(float(h[tr,tc])+3+dz[lo:lo+8192,None]*t)
  out.append(torch.asinh(e/2)/3)
 return torch.cat(out)
@torch.inference_mode()
def surface_gpu(h,fields,vis,tr,tc):
 device=h.device;n=65536
 td=lambda a:torch.as_tensor(a,device=device,dtype=torch.float64)
 yy,xx=torch.meshgrid(torch.arange(256,device=device,dtype=torch.float64),torch.arange(256,device=device,dtype=torch.float64),indexing='ij')
 rr=yy.flatten();cc=xx.flatten();zrx=h.flatten()+1
 rad=td(RADII)[None,:];area=td(AREA)[None,:]
 fd=td(fields)[None];vd=td(np.concatenate(vis,axis=0))[None]
 acc=[torch.zeros((n,21),device=device,dtype=torch.float64) for _ in range(2)]
 ang=2*np.pi*np.arange(16)/16;ht=float(h[tr,tc])
 for a in range(16):
  ox=td(np.cos(ang[a])*RADII)[None,:];oy=td(np.sin(ang[a])*RADII)[None,:]
  qx=cc[:,None]+ox;qy=rr[:,None]+oy;inside=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255)
  grid=torch.stack((qx/127.5-1,qy/127.5-1),-1)[None]
  hh,sx,sy=F.grid_sample(fd,grid,align_corners=True,padding_mode='border')[0].float().unbind(0)
  vv=F.grid_sample(vd,grid,align_corners=True,padding_mode='zeros')[0].float()
  norm=torch.sqrt(1+sx*sx+sy*sy)
  rz=zrx[:,None]-(hh+.05);rlen=torch.sqrt(rad.square()+rz.square())
  coso=((sx*ox+sy*oy+rz)/norm/rlen).clamp_min(0)
  slope=(hh-zrx[:,None])/rad
  prior=torch.cat((torch.full((n,1),-1e6,device=device,dtype=torch.float64),slope.cummax(1).values[:,:-1]),1)
  rxvis=(slope+.05/rad>=prior)&inside
  for j,txheight in enumerate([3,2]):
   dst=acc[j];txz=ht+txheight-(hh+.05);txx=tc-qx;txy=tr-qy
   tlen=torch.sqrt(txx.square()+txy.square()+txz.square()).clamp_min(.25)
   cosi=((-sx*txx-sy*txy+txz)/norm/tlen).clamp_min(0)
   dz=zrx-ht-txheight;d0sq=(cc-tc).square()+(rr-tr).square()+dz.square()
   geom=area*norm*cosi*coso/(tlen.square()*rlen.square())*d0sq[:,None]/np.pi
   vis0=rxvis*vv[j*3];w=geom*vis0;wl=geom*rxvis*vv[j*3+1];wu=geom*rxvis*vv[j*3+2]
   extra=(tlen+rlen-torch.sqrt(d0sq)[:,None]).clamp_min(0)
   align=((-txx*(-ox)-txy*(-oy)-txz*rz)/(tlen*rlen)+2*cosi*coso).clamp(-1,1)
   total=w.sum(1)
   dst[:,0]+=total;dst[:,1]+=w[:,:5].sum(1);dst[:,2]+=w[:,5:10].sum(1);dst[:,3]+=w[:,10:].sum(1);dst[:,4]+=wl.sum(1);dst[:,5]+=wu.sum(1)
   mx,ix=w.max(1);better=mx>dst[:,6];dst[:,6]=torch.maximum(dst[:,6],mx)
   dst[:,18]=torch.where(better,extra.gather(1,ix[:,None])[:,0],dst[:,18]);dst[:,19]=torch.where(better,rad[0,ix],dst[:,19])
   for k,val in [(7,extra),(8,rad),(9,tlen),(10,cosi),(11,coso),(12,align)]:
    dst[:,k]+=(w*val).sum(1)
   dst[:,13]+=(w*align.clamp_min(0).pow(32)).sum(1);dst[:,14]+=total*np.cos(ang[a]);dst[:,15]+=total*np.sin(ang[a])
   dst[:,16]+=vis0.sum(1);dst[:,17]+=(vis0*area*norm).sum(1);dst[:,20]+=inside.sum(1)
 result=[]
 for dst in acc:
  total=dst[:,0];den=total.clamp_min(1e-20);lg=lambda v:torch.log10(v+1e-16)
  vals=[lg(dst[:,k]) for k in range(7)]+[torch.log1p(dst[:,k]/den) for k in [7,8,9]]+[dst[:,k]/den for k in [10,11,12]]+[lg(dst[:,13])]+[dst[:,k]/den for k in [14,15]]+[dst[:,16]/dst[:,20].clamp_min(1),torch.log1p(dst[:,17]),torch.where(total>0,torch.log1p(dst[:,18]),0),torch.where(total>0,torch.log1p(dst[:,19]),0)]
  result.extend(vals)
 return torch.stack(result,1).float()

