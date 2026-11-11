"""Localize rare direct deficits on a native receiver mesh before predicting their two-band magnitude."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1]

class Inputs:
    def __init__(self):
        p=np.load(R/'src/c70_centroid_lookup0.npz')
        self.ids=torch.from_numpy(p['ids'].astype(np.int64)).cuda()
        self.weights=torch.from_numpy(p['weights'].astype(np.float32)).cuda()
        cells=torch.arange(65025,device='cuda')%65025
        self.cx=torch.cat((cells%255+2/3,cells%255+1/3))
        self.cy=torch.cat((cells//255+1/3,cells//255+2/3))
    def __call__(self,h,rows,cols,pred,compact):
        B=len(rows);device=h.device
        a=h[:,:-1,:-1].flatten(1);b=h[:,:-1,1:].flatten(1);c=h[:,1:,:-1].flatten(1);d=h[:,1:,1:].flatten(1)
        sx=torch.cat((b-a,d-c),1);sy=torch.cat((d-b,c-a),1)
        alt=torch.cat(((a+b+d)/3,(a+c+d)/3),1)+1
        z=h[torch.arange(B,device=device),(rows+1).clamp(0,255),cols].floor()+3
        xl=self.cx[None]-cols[:,None]*255/256;yl=self.cy[None]-(rows[:,None]+1)*255/256
        dx=xl*(256/255);dy=yl*(256/255);dz=alt-z[:,None]
        rxy=torch.sqrt(dx.square()+dy.square()).clamp_min(.25);dist=torch.sqrt(rxy.square()+dz.square()).clamp_min(.25)
        s2=(sx.square()+sy.square())*(255/256)**2
        plane=dz-xl*sx-yl*sy
        cos=plane/(dist*torch.sqrt(1+s2))
        transverse=(sx*dy-sy*dx)/rxy;parallel=(sx*dx+sy*dy)/rxy
        hits=4e9/(4*np.pi)*.5*(256/255)**2*torch.sqrt(1+s2)*abs(cos)/dist.square()
        # Retain each face separately. Sorted weights avoid arbitrary Qhull simplex order.
        order=self.weights.argsort(dim=1,descending=True)
        idx=self.ids.gather(1,order)
        w=self.weights.gather(1,order)
        def gather(v):return v[:,idx].permute(0,2,1)
        face=torch.cat([gather(t) for t in (torch.asinh(plane*2)/4,torch.asinh(cos*140)/4,
            torch.log10(abs(cos).clamp_min(1e-6))/4,torch.log1p(hits)/8,1/(1+hits/10),
            parallel.clamp(-2,2),transverse.clamp(-2,2))],1)
        distmap=((dist[:,idx]*w).sum(2)).clamp_min(.25).reshape(B,256,256)
        fs=20*torch.log10(distmap[:,None]*torch.tensor([415e6,5.775e9],device=device)[None,:,None,None]*(4*np.pi/299792458))
        yy,xx=torch.meshgrid(torch.arange(256,device=device),torch.arange(256,device=device),indexing='ij')
        rx=xx[None]-cols[:,None,None];ry=yy[None]-rows[:,None,None];dist2=torch.sqrt(rx.square()+ry.square()).clamp_min(1)
        theta=torch.atan2(ry,rx);s=h-h[torch.arange(B,device=device),rows,cols][:,None,None]
        image=torch.cat([(pred-fs)/12,(pred[:,1:2]-pred[:,:1]-20*np.log10(5.775e9/415e6))/10,
             fs[:,:1]/80,s[:,None]/40,sx[:,:65025].reshape(B,1,255,255).mean((2,3),keepdim=True).expand(-1,-1,256,256),
             (rx/dist2)[:,None],(ry/dist2)[:,None],torch.sin(2*theta)[:,None],torch.cos(2*theta)[:,None],torch.log1p(dist2)[:,None]/5,
             face.reshape(B,-1,256,256), w.T[None].expand(B,-1,-1).reshape(B,3,256,256),
             compact[:,[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,27,28,31,32,33,34,39,40,41,42,43,44]]],1)
        assert torch.isfinite(image).all()
        return image.clamp(-12,12),fs
class Block(nn.Module):
    def __init__(self,n,dilation):
        super().__init__();self.branch=nn.Sequential(nn.GroupNorm(4,n),nn.Conv2d(n,n,3,padding=dilation,dilation=dilation,groups=n),
            nn.Conv2d(n,n*2,1),nn.GELU(),nn.Conv2d(n*2,n,1))
    def forward(self,x):return x+self.branch(x)

class WeakField(nn.Module):
    def __init__(self,num_channels,width=40):
        super().__init__()
        self.stem=nn.Conv2d(num_channels,width,1)
        self.local=nn.Sequential(Block(width,1),Block(width,2),Block(width,1))
        self.coarse=nn.Sequential(Block(width,1),Block(width,2),Block(width,4))
        self.merge=nn.Sequential(nn.Conv2d(2*width,width,1),nn.GELU(),Block(width,1),nn.Conv2d(width,6,1))
        nn.init.zeros_(self.merge[-1].weight);nn.init.zeros_(self.merge[-1].bias)
    def forward(self,x,pred,fs):
        v=self.local(self.stem(x));ctx=self.coarse(F.avg_pool2d(v,4))
        g=self.merge(torch.cat([v,F.interpolate(ctx,size=v.shape[-2:],mode='bilinear',align_corners=False)],1)).float()
        logprior=(pred-fs-3)/2.5;prior=logprior.clamp(-15,15).sigmoid()
        logit=logprior.clamp(-15,15)+g[:,:2]*2
        gap=6.;strong=pred-gap*prior+4*g[:,2:4];weak=pred+gap*(1-prior)+4*g[:,4:]
        pp=logit.sigmoid()
        return (1-pp)*strong+pp*weak,logit,strong,weak
