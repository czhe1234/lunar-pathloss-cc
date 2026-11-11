"""C4 adds exact piecewise-linear below-ground lengths and critical endpoint heights."""
import torch
from torch.nn import functional as F
from c3_geometry import geometry,grid_base,basic_inputs

@torch.no_grad()
def extra_geometry(h,r,c):
    h=h.float();b=len(h);yy,xx=grid_base(h);ht=h[torch.arange(b,device=h.device),r,c]
    dx=xx[None]-c[:,None,None];dy=yy[None]-r[:,None,None];dz=h+1-ht[:,None,None]-3
    needtx=torch.full_like(h,-1e6);needrx=torch.full_like(h,-1e6);len3=torch.zeros_like(h);len2=torch.zeros_like(h)
    for k in range(8):
        t=torch.arange(k*32,k*32+33,device=h.device,dtype=torch.float32)/256
        px=(c[:,None,None,None]+dx[:,None]*t[None,:,None,None])/127.5-1
        py=(r[:,None,None,None]+dy[:,None]*t[None,:,None,None])/127.5-1
        grid=torch.stack((px,py),-1).reshape(b,33*256,256,2)
        hh=F.grid_sample(h[:,None],grid,padding_mode='border',align_corners=True)[:,0].reshape(b,33,256,256)
        e=hh-(ht[:,None,None,None]+3+dz[:,None]*t[None,:,None,None])
        tx=e/(1-t).clamp_min(1e-6)[None,:,None,None];tx=tx.masked_fill((t==1)[None,:,None,None],-1e6)
        rx=e/t.clamp_min(1e-6)[None,:,None,None];rx=rx.masked_fill((t==0)[None,:,None,None],-1e6)
        needtx=torch.maximum(needtx,tx.amax(1));needrx=torch.maximum(needrx,rx.amax(1))
        for vv,total in [(e,len3),(e+(1-t)[None,:,None,None],len2)]:
            a=vv[:,:-1];z=vv[:,1:]
            fraction=(a.clamp_min(0)+z.clamp_min(0))/(a.abs()+z.abs()).clamp_min(1e-7)
            total+=fraction.sum(1)/256
    distance3=torch.sqrt(dx.square()+dy.square()+dz.square());distance2=torch.sqrt(dx.square()+dy.square()+(dz+1).square())
    g=torch.stack((torch.asinh((needtx+3-2.5)/2)/3,torch.asinh(needrx/2)/3,torch.log1p(len3*distance3)/6,torch.log1p(len2*distance2)/6),1)
    return g.half()

def make_inputs(h,r,c,features=None):
    x,fs=basic_inputs(h,r,c)
    if features is None:features=torch.cat((geometry(h,r,c),extra_geometry(h,r,c)),1)
    return torch.cat((x,features.float()),1).contiguous(memory_format=torch.channels_last),fs
