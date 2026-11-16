"""C32: terrain-query model with two deterministic C10 prediction channels."""
import math
from c21_model_r1 import SpatialCenterNorm
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Module):
 def __init__(self,ci,co):
  super().__init__();self.skip=nn.Conv2d(ci,co,1,bias=False) if ci!=co else nn.Identity()
  self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.SiLU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(8,co))
 def forward(self,x):return F.silu(self.net(x)+self.skip(x))
class PointRes(nn.Module):
 def __init__(self,w):
  super().__init__();self.a=nn.Linear(w,w);self.b=nn.Linear(w,w)
 def forward(self,x):return F.silu(x+.2*self.b(F.silu(self.a(x))))
class Model(nn.Module):
 def __init__(self,mean,std,geometry=True,width=256,layers=4):
  super().__init__();self.geometry=geometry;self.width=width;self.heads=8
  self.register_buffer('mean',mean[None,:,None,None]);self.register_buffer('std',std[None,:,None,None])
  self.encoder=nn.ModuleList([Block(47,64),Block(64,128),Block(128,192),Block(192,width)])
  self.localproj=nn.ModuleList([nn.Conv2d(c,32,1) for c in [128,192,width]])
  self.local=Block(160,96)
  self.position=nn.Linear(8,width);self.scale=nn.Parameter(torch.zeros(3,width))
  self.scene=nn.ModuleList([nn.TransformerEncoderLayer(width,8,width*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
  self.freq=nn.Parameter(torch.randn(2,16)*.02)
  self.query=nn.Linear(96+47+64+16,width);self.qnorm=nn.LayerNorm(width);self.knorm=nn.LayerNorm(width)
  self.knorm=SpatialCenterNorm(width)
  self.kv=nn.Linear(width,2*width);self.attout=nn.Linear(width,width)
  self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
  self.geoweight=nn.Parameter(torch.zeros(2,8,8),requires_grad=bool(geometry))
  self.readout=nn.ModuleList([nn.Sequential(nn.Linear(96+47+64+width,512),nn.SiLU(),PointRes(512),PointRes(512),nn.Linear(512,1)) for _ in range(2)])
  for head in self.readout:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
  yy,xx=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
  self.register_buffer('xy',torch.stack((yy,xx)),persistent=False);self.register_buffer('frequencies',2*math.pi*2**torch.arange(8,dtype=torch.float32),persistent=False)
 def encode(self,x,tx):
  raw=x.float();z=((raw-self.mean)/self.std).clamp(-8,8);features=[];h=z
  for i,block in enumerate(self.encoder):
   if i:h=F.avg_pool2d(h,2)
   h=block(h);features.append(h)
  loc=self.local(torch.cat((features[0],*[F.interpolate(p(f),size=(256,256),mode='bilinear',align_corners=False) for p,f in zip(self.localproj,features[1:])]),1))
  tokens=[];geom=[];B=len(x)
  maps=torch.cat((self.xy[None].expand(B,-1,-1,-1),raw[:,0:1]*100,raw[:,3:5]*5),1)
  for i,size in enumerate([32,16,8]):
   t=F.adaptive_avg_pool2d(features[-1],size).flatten(2).transpose(1,2)
   g=F.adaptive_avg_pool2d(maps,size).flatten(2).transpose(1,2)
   dr=g[:,:,0]-tx[:,None,0];dc=g[:,:,1]-tx[:,None,1];dz=g[:,:,2]-tx[:,None,2];dd=torch.sqrt(dr*dr+dc*dc+dz*dz).clamp_min(.25)
   pos=torch.stack((g[:,:,0]/128-1,g[:,:,1]/128-1,g[:,:,2]/100,dr/128,dc/128,dz/50,torch.log1p(dd)/6,torch.ones_like(dd)*i/2),-1)
   tokens.append(t+self.position(pos)+self.scale[i]);geom.append(g)
  t=torch.cat(tokens,1)
  for block in self.scene:t=block(t)
  kv=self.kv(self.knorm(t)).reshape(B,-1,2,self.heads,self.width//self.heads).permute(2,0,3,1,4)
  return loc,z,torch.cat(geom,1),kv,tx,raw
 def geometry_features(self,pix,geo,tx,raw):
  rr=(pix//256).float();cc=(pix%256).float();hh=raw[:,0].flatten(1).gather(1,pix)*100+1
  sr=geo[:,None,:,0]-rr[:,:,None];sc=geo[:,None,:,1]-cc[:,:,None];sz=geo[:,None,:,2]-hh[:,:,None]
  ds=torch.sqrt(sr*sr+sc*sc+sz*sz+.0625)
  tr=tx[:,None,0]-geo[:,:,0];tc=tx[:,None,1]-geo[:,:,1];tz=tx[:,None,2]-geo[:,:,2]
  dt=torch.sqrt(tr*tr+tc*tc+tz*tz+.0625)
  direct=torch.sqrt((rr-tx[:,None,0])**2+(cc-tx[:,None,1])**2+(hh-tx[:,None,2])**2+.0625)
  nx=-geo[:,:,3];ny=-geo[:,:,4];norm=torch.sqrt(1+nx*nx+ny*ny)
  ct=(nx*tc+ny*tr+tz)/norm/dt
  cr=(-nx[:,None]*sc-ny[:,None]*sr-sz)/norm[:,None]/ds
  return torch.stack((sr/128,sc/128,sz/50,torch.log1p(ds)/6,torch.log1p(dt)[:,None].expand_as(ds)/6,
    torch.log1p((ds+dt[:,None]-direct[:,:,None]).clamp_min(0))/6,ct[:,None].expand_as(ds),cr),-1).clamp(-8,8)
 def decode(self,state,pixels,chunk=256):
  loc,z,geo,kv,tx,raw=state;B=len(z);out=[]
  for lo in range(0,pixels.shape[1],chunk):
   pix=pixels[:,lo:lo+chunk];Q=pix.shape[1]
   local=loc.flatten(2).gather(2,pix[:,None].expand(-1,loc.shape[1],-1)).transpose(1,2)
   orig=z.flatten(2).gather(2,pix[:,None].expand(-1,47,-1)).transpose(1,2)
   rr=(pix//256).float();cc=(pix%256).float()
   coords=torch.stack((rr/256,cc/256,(rr-tx[:,None,0])/256,(cc-tx[:,None,1])/256),-1)
   angle=coords[:,:,:,None]*self.frequencies;pos=torch.cat((angle.sin(),angle.cos()),-1).flatten(2)
   g=self.geometry_features(pix,geo,tx,raw) if self.geometry else None
   bands=[]
   for b in [0,1]:
    q=self.query(torch.cat((local,orig,pos,self.freq[b][None,None].expand(B,Q,-1)),-1));qq=self.qnorm(q).reshape(B,Q,self.heads,-1).transpose(1,2)
    bias=None
    if self.geometry:
     with torch.autocast('cuda',enabled=False):bias=torch.einsum('bqkf,hf->bhqk',g,self.geoweight[b].float())
     bias=bias.to(qq.dtype)
    attended=F.scaled_dot_product_attention(qq,kv[0],kv[1],attn_mask=bias,dropout_p=0.)
    q=q+self.attout(attended.transpose(1,2).reshape(B,Q,-1));q=q+self.ff(q)
    hidden=self.readout[b][:-1](torch.cat((local,orig,pos,q),-1))
    with torch.autocast('cuda',enabled=False):pred=self.readout[b][-1](hidden.float()).squeeze(-1)
    bands.append(pred)
   out.append(torch.stack(bands,-1))
  return torch.cat(out,1)
 def forward(self,x,tx,pixels,chunk=256):return self.decode(self.encode(x,tx),pixels,chunk)
