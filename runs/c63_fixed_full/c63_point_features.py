"""Batched arbitrary-pixel C38 descriptors. Only released H/TX-derived caches."""
import math,csv,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from src.c47_train import device_array
from src.c49_train import terrain_fields
from src.c10_phase_geometry import phase_features_gpu
from src.c7_surface_features import RADII,AREA
R=Path(__file__).resolve().parents[1]
class FeatureBank:
 def __init__(self):
  self.fields,self.info=terrain_fields();self.g3=np.load(R/'features/c3/train.npy',mmap_mode='r');self.g4=np.load(R/'features/c4_extra/train.npy',mmap_mode='r')
  n=len(self.info);self.vis=torch.empty((n,6,256,256),device='cuda',dtype=torch.float32)
  for lo in range(0,n,64):
   bins=torch.from_numpy(np.array(self.g3[lo:lo+64,:8],np.float32)).cuda()
   em3,em2=self.margins(bins);vv=[]
   for em,tol in [(em3,.95/16),(em2,.05/16)]:
    vv.extend([1/(1+torch.exp((a/.03).clamp(-60,60))) for a in [em,em+tol,em-tol]])
   self.vis[lo:lo+len(bins)]=torch.stack(vv,1)
  del bins,vv,em3,em2;torch.cuda.empty_cache()
  context=np.zeros((n,73),np.float32)
  for part,folder,lab,P in [('fit','c38_dense','c38_dense',1024),('dev','c10_points','c6_points',256)]:
   ids=np.load(R/f'features/{lab}/{part}_map_indices.npy');x=np.load(R/f'features/{folder}/{part}_x.npy',mmap_mode='r');context[ids]=x[np.arange(len(ids))*P,31:104]
  self.context=torch.from_numpy(context).cuda()
  yy,xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij');self.yy=yy;self.xx=xx
  self.rad=torch.tensor(RADII,device='cuda',dtype=torch.float64)[None,None,None]
  self.area=torch.tensor(AREA,device='cuda',dtype=torch.float64)[None,None,None]
  a=2*np.pi*np.arange(16)/16
  self.ox=torch.tensor(np.cos(a)[:,None]*RADII,device='cuda',dtype=torch.float64)[None,None]
  self.oy=torch.tensor(np.sin(a)[:,None]*RADII,device='cuda',dtype=torch.float64)[None,None]
  self.ca=torch.tensor(np.cos(a),device='cuda',dtype=torch.float64)[None,None,:,None]
  self.sa=torch.tensor(np.sin(a),device='cuda',dtype=torch.float64)[None,None,:,None]
 @staticmethod
 def margins(bins):
  t=(torch.arange(8,device=bins.device,dtype=torch.float32)+.5)/8;e=2*torch.sinh(3*bins.float())
  return (e+.95*t[None,:,None,None]).amax(1),(e+.95*t[None,:,None,None]+(1-t[None,:,None,None])).amax(1)
 @torch.no_grad()
 def image(self,maps):
  inf=self.info[maps];B=len(maps);field=self.fields[inf[:,0]];h=field[...,0];sx=field[...,1];sy=field[...,2];tr=inf[:,1];tc=inf[:,2];ht=h[torch.arange(B,device='cuda'),tr,tc]
  dx=self.xx[None]-tc[:,None,None];dy=self.yy[None]-tr[:,None,None];dz=h+1-ht[:,None,None]-3;dist=torch.sqrt(dx.square()+dy.square()+dz.square()).clamp_min(.25)
  hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0]
  bins=torch.from_numpy(np.array(self.g3[maps.cpu().numpy(),:8],np.float32)).cuda();em3,em2=self.margins(bins);em3=em3.half().float();em2=em2.half().float()
  vals=[h/100,(h-ht[:,None,None])/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(dist)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx.square()+dy.square())/8),torch.asinh(em3/2)/3,torch.asinh(em2/2)/3]
  return torch.stack(vals,1).contiguous(memory_format=torch.channels_last)
 @torch.no_grad()
 def surface(self,maps,pixels):
  B,Q=pixels.shape;inf=self.info[maps];hm=inf[:,0];tr=inf[:,1];tc=inf[:,2]
  rr=(pixels//256).double()[:,:,None,None];cc=(pixels%256).double()[:,:,None,None]
  qx=cc+self.ox;qy=rr+self.oy;inside=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255);grid=torch.stack((qx/127.5-1,qy/127.5-1),-1).reshape(B,Q,256,2)
  fd=self.fields[hm].permute(0,3,1,2).double();v=self.vis[maps].double()
  hh,sx,sy=F.grid_sample(fd,grid,align_corners=True,padding_mode='border').permute(0,2,3,1).reshape(B,Q,16,16,3).float().unbind(-1)
  vv=F.grid_sample(v,grid,align_corners=True,padding_mode='zeros').permute(0,2,1,3).reshape(B,Q,6,16,16).float()
  ht=self.fields[hm,tr,tc,0];rz0=self.fields[hm[:,None],pixels//256,pixels%256,0]+1
  norm=torch.sqrt(1+sx.square()+sy.square());rz=rz0[:,:,None,None]-(hh+.05);dr=torch.sqrt(self.rad.square()+rz.square())
  co=((sx*self.ox+sy*self.oy+rz)/norm/dr).clamp_min(0);slope=(hh-rz0[:,:,None,None])/self.rad
  prior=F.pad(slope.cummax(-1).values[...,:-1],(1,0),value=-1e6);rxvis=(slope+.05/self.rad>=prior)&inside
  result=[];sm=lambda a:a.sum((-2,-1))
  for j,height in enumerate([3,2]):
   tz=ht[:,None,None,None]+height-(hh+.05);tx=tc[:,None,None,None]-qx;ty=tr[:,None,None,None]-qy
   dt=torch.sqrt(tx.square()+ty.square()+tz.square()).clamp_min(.25);ci=((-sx*tx-sy*ty+tz)/norm/dt).clamp_min(0)
   dz=rz0-ht[:,None]-height;d2=(pixels%256-tc[:,None]).double().square()+(pixels//256-tr[:,None]).double().square()+dz.square()
   geom=self.area*norm*ci*co/(dt.square()*dr.square())*d2[:,:,None,None]/math.pi
   w=geom*rxvis*vv[:,:,j*3];wl=geom*rxvis*vv[:,:,j*3+1];wu=geom*rxvis*vv[:,:,j*3+2]
   ex=(dt+dr-torch.sqrt(d2)[:,:,None,None]).clamp_min(0);align=((-tx*(-self.ox)-ty*(-self.oy)-tz*rz)/(dt*dr)+2*ci*co).clamp(-1,1)
   total=sm(w);den=total.clamp_min(1e-20);mean=lambda a:sm(w*a)/den;lg=lambda a:torch.log10(a+1e-16)
   flat=w.flatten(2);mx,best=flat.max(-1);bx=ex.flatten(2).gather(-1,best[:,:,None])[:,:,0];br=self.rad.flatten()[best%16]
   vals=[lg(total),lg(sm(w[...,:5])),lg(sm(w[...,5:10])),lg(sm(w[...,10:])),lg(sm(wl)),lg(sm(wu)),lg(mx),torch.log1p(mean(ex)),torch.log1p(mean(self.rad)),torch.log1p(mean(dt)),mean(ci),mean(co),mean(align),lg(sm(w*align.clamp_min(0).pow(32))),mean(self.ca),mean(self.sa),sm(rxvis*vv[:,:,j*3])/inside.sum((-1,-2)).clamp_min(1),torch.log1p(sm(rxvis*vv[:,:,j*3]*self.area*norm)),torch.where(total>0,torch.log1p(bx),0),torch.where(total>0,torch.log1p(br),0)]
   result.extend(vals)
  return torch.stack(result,-1).float()
 @torch.no_grad()
 def points(self,maps,pixels,image=None):
  if image is None:image=self.image(maps)
  B,Q=pixels.shape;basic=image[:,:11].flatten(2).gather(2,pixels[:,None].expand(-1,11,-1)).transpose(1,2)
  mid=maps.cpu().numpy();p=pixels.cpu().numpy();rr=p//256;cc=p%256
  gg=np.concatenate((self.g3[mid[:,None],:,rr,cc],self.g4[mid[:,None],:,rr,cc]),-1).astype(np.float32)
  assert gg.shape==(B,Q,20),gg.shape;g=torch.from_numpy(gg).cuda()
  context=self.context[maps,None].expand(-1,Q,-1);first=torch.cat((basic,g,context),-1)
  inf=self.info[maps];tr=inf[:,1].float();tc=inf[:,2].float();ht=context[:,0,0]
  rx=(pixels%256).float();ry=(pixels//256).float();dx=rx-tc[:,None];dy=ry-tr[:,None];dz=basic[:,:,7]*50
  t=(torch.arange(128,device='cuda',dtype=torch.float32)+.5)/128
  gx=tc[:,None,None]+dx[:,:,None]*t;gy=tr[:,None,None]+dy[:,:,None]*t
  grid=torch.stack((gx.double()/127.5-1,gy.double()/127.5-1),-1)
  hh=F.grid_sample(self.fields[inf[:,0],:,:,0].double()[:,None],grid,align_corners=True,padding_mode='border')[:,0].float()
  # dz follows the original profile's chained raw arithmetic rather than round-trip normalization.
  zrx=self.fields[inf[:,0,None],pixels//256,pixels%256,0];pdz=zrx+1-ht[:,None]-3
  e=hh-(ht[:,None,None]+3+pdz[:,:,None]*t);profile=torch.asinh(e/2)/3
  surface=self.surface(maps,pixels)
  crit=2.5+2*torch.sinh(3*first[:,:,27]);phase=torch.arange(8,device='cuda',dtype=torch.float64)/8
  extra=torch.asinh((crit.double()[:,:,None]-(3-torch.remainder(ht.double()[:,None,None]+phase,1)))/2)/3
  x=torch.cat((first,profile,surface,extra.float()),-1);ph=phase_features_gpu(x.reshape(-1,280)).reshape(B,Q,14);x=torch.cat((x,ph),-1)
  dist=torch.sqrt(dx.double().square()+dy.double().square()+pdz.double().square()).clamp_min(.25)
  freq=torch.tensor([415e6,5.775e9],device='cuda',dtype=torch.float64)
  fspl=(20*torch.log10(4*math.pi*dist[:,:,None]*freq/299792458.)).float()
  assert x.shape==(B,Q,294) and torch.isfinite(x).all();return x,fspl
