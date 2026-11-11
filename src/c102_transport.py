"""Receiver-to-surface cross attention with soft geometric power weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F
class Residual(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(n),nn.Linear(n,2*n),nn.SiLU(),nn.Linear(2*n,n))
 def forward(self,x):return x+.25*self.net(x)
class Transport(nn.Module):
 def __init__(self):
  super().__init__();self.root=nn.Sequential(nn.Linear(681,128),nn.SiLU(),Residual(128),Residual(128))
  self.nodes=nn.Sequential(nn.Linear(106,64),nn.SiLU(),Residual(64))
  self.q=nn.Linear(128,32);self.kv=nn.Linear(64,64)
  self.head=nn.Sequential(nn.Linear(161,128),nn.SiLU(),Residual(128),nn.Linear(128,2))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,query,nodes,bias,mass):
  b,q,_=query.shape;root=self.root(query);node=self.nodes(nodes);k,v=self.kv(node).chunk(2,-1)
  qq=self.q(root).reshape(b,q,4,8).transpose(1,2)
  kk=k.reshape(b,-1,4,8).transpose(1,2);vv=v.reshape(b,-1,4,8).transpose(1,2)
  ctx=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=bias[:,None].to(qq.dtype),dropout_p=0.)
  ctx=ctx.transpose(1,2).reshape(b,q,32)
  return 4*self.head(torch.cat([root,ctx,mass[:,:,None].to(root.dtype)/8],-1)).float()
@torch.no_grad()
def geometry(h,tr,tc,pixels,compact,scale):
 b=len(h);dev=h.device;n=256//scale
 sy,sx=torch.gradient(h,dim=(-2,-1))
 fields=torch.stack([h,sx,sy,compact[:,16].clamp(0,1)],1)
 if scale>1:fields=F.avg_pool2d(fields,scale)
 hs,sx,sy,vis=fields.flatten(2).unbind(1);normal=torch.stack([-sx,-sy,torch.ones_like(sx)],-1)
 normal=normal/normal.square().sum(-1,keepdim=True).sqrt()
 axis=torch.arange(n,device=dev,dtype=torch.float32)*scale+(scale-1)/2
 yy,xx=torch.meshgrid(axis,axis,indexing='ij');xx=xx.reshape(1,-1);yy=yy.reshape(1,-1)
 tz=h[torch.arange(b,device=dev),tr,tc]+3
 txd=torch.stack([tr[:,None]*0+tc[:,None]-xx,tr[:,None]-yy,tz[:,None]-hs],-1)
 dt=txd.square().sum(-1).clamp_min(.25).sqrt();td=txd/dt[:,:,None]
 ci=(td*normal).sum(-1);reflection=-td+2*ci[:,:,None]*normal
 rr=pixels//256;cc=pixels%256;rz=h[torch.arange(b,device=dev)[:,None],rr,cc]+1
 dx=cc.float()[:,:,None]-xx[:,None];dy=rr.float()[:,:,None]-yy[:,None];dz=rz[:,:,None]-hs[:,None]
 dr=(dx*dx+dy*dy+dz*dz).clamp_min(.25).sqrt()
 co=(dx*normal[:,None,:,0]+dy*normal[:,None,:,1]+dz*normal[:,None,:,2])/dr
 align=(dx*reflection[:,None,:,0]+dy*reflection[:,None,:,1]+dz*reflection[:,None,:,2])/dr
 softci=F.softplus(ci*16)/16;softco=F.softplus(co*16)/16
 kernel=(softci[:,None]+.005).log()+(softco+.005).log()+(vis[:,None]+.01).log()-2*dt[:,None].log()-2*dr.log()
 kernel+=torch.log(.15+.85*torch.exp(24*(align.clamp(max=1)-1)))+math.log(scale*scale)
 mass=torch.logsumexp(kernel,2)/math.log(10)
 bias=kernel-kernel.amax(2,keepdim=True)
 return bias,mass
@torch.no_grad()
def frozen_field(net,x):
 b=x.shape[0];r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
 x=x.clamp(-12,12);tx=x[torch.arange(b,device=x.device),:,r,c]
 fine=net.native(net.stem(x));mid=net.stage1(net.condition(net.down1(fine),tx,net.film1))
 coarse=net.stage2(net.condition(net.down2(mid),tx,net.film2))
 mid=net.decode1(mid+F.interpolate(net.up1(coarse),size=mid.shape[-2:],mode='bilinear',align_corners=False))
 return net.decode0(fine+F.interpolate(net.up0(mid),size=fine.shape[-2:],mode='bilinear',align_corners=False))
@torch.no_grad()
def frozen_query(net,x,precise):
 z=net.embedding(x)[:,None].expand(-1,8,-1)
 for layer in net.layers:z=F.relu(layer(z))
 with torch.autocast('cuda',enabled=False):
  delta=(torch.einsum('nkw,kwo->nko',z.float(),net.head.float())+net.bias.float()).mean(1)*20
  raw=((precise-net.embedding.mean)/net.embedding.std).clamp(-20,20)
 return torch.cat([z.mean(1).float(),raw],1),delta
