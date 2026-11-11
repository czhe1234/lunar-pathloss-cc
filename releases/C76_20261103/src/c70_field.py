"""Multiscale field model with explicit sampling along and beside the TX-to-RX direction."""
import torch
from torch import nn
from torch.nn import functional as F

class StageBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.spatial=nn.Conv2d(c,c,5,padding=2,groups=c)
        self.norm=nn.LayerNorm(c);self.up=nn.Linear(c,2*c);self.down=nn.Linear(2*c,c)
        self.alpha=nn.Parameter(torch.full((c,),.15))
    def forward(self,x):
        v=self.spatial(x).permute(0,2,3,1)
        v=self.down(F.gelu(self.up(self.norm(v))))*self.alpha
        return x+v.permute(0,3,1,2)
class Corridor(nn.Module):
    def __init__(self,c):
        super().__init__();self.kv=nn.Conv2d(c,32+48,1);self.q=nn.Conv2d(c,32,1)
        self.out=nn.Conv2d(48,c,1)
        points=[(a,b) for a in (0.,.12,.3,.5,.7,.88,1.02) for b in ([-.12,0.,.12] if a>0 else [0.])]
        self.register_buffer('ray_points',torch.tensor(points,dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(len(points),4))
    def forward(self,x,r,c):
        b,_,h,w=x.shape;dev=x.device
        axis=torch.linspace(-1,1,h,device=dev,dtype=torch.float32)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        ty=r[:,None,None,None];tx=c[:,None,None,None]
        dy=yy[None,None]-ty;dx=xx[None,None]-tx
        u=self.ray_points[:,0,None,None];v=self.ray_points[:,1,None,None]
        gx=tx+u*dx-v*dy;gy=ty+u*dy+v*dx
        grid=torch.stack((gx,gy),-1).reshape(b,-1,w,2)
        # Four heads see the same candidate geometry and assign different source weights.
        n=len(self.ray_points)
        z=F.grid_sample(self.kv(x).float(),grid,padding_mode='border',align_corners=True).reshape(b,80,n,h,w)
        key=z[:,:32].reshape(b,4,8,n,h,w);value=z[:,32:].reshape(b,4,12,n,h,w)
        q=self.q(x).float().reshape(b,4,8,1,h,w)
        scores=(key*q).sum(2)/8**.5+self.bias.T[None,:,:,None,None]
        scores=scores.masked_fill(((gx.abs()>1)|(gy.abs()>1))[:,None],-30.)
        context=(scores.softmax(2)[:,:,None]*value).sum(3).reshape(b,48,h,w)
        return x+self.out(context)
class FieldMap(nn.Module):
    def __init__(self,inputs,width=40):
        super().__init__()
        c=[width,width+8,width+24,width+48,width+72]
        self.stem=nn.Conv2d(inputs,c[0],1)
        self.stage=nn.ModuleList([nn.Sequential(*(StageBlock(k) for _ in range(2 if j>=2 else 1))) for j,k in enumerate(c)])
        self.down=nn.ModuleList([nn.Conv2d(a,b,3,stride=2,padding=1) for a,b in zip(c,c[1:])])
        self.up=nn.ModuleList([nn.Conv2d(c[j+1],c[j],1) for j in range(4)])
        self.dec=nn.ModuleList([StageBlock(c[j]) for j in range(4)])
        self.corridor=Corridor(c[-1]);self.head=nn.Conv2d(c[0],2,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        # Input channels 5 and 6 are signed column and row displacements / 128.
        tc=(-x[:,5,0,0].float()*128)/127.5-1;tr=(-x[:,6,0,0].float()*128)/127.5-1
        v=self.stage[0](self.stem(x));skip=[v]
        for j,down in enumerate(self.down):
            v=self.stage[j+1](down(v));skip.append(v)
        v=self.corridor(v,tr,tc)
        for j in (3,2,1,0):
            v=self.dec[j](skip[j]+F.interpolate(self.up[j](v),size=skip[j].shape[-2:],mode='bilinear',align_corners=False))
        return 4*self.head(v)
