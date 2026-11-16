"""C3 input-only geometric descriptors; no propagation simulation or labels."""
import math
import torch
from torch.nn import functional as F
N_GEOM=16
PROFILE_STEPS=256
RADII=(2.,4.,6.,8.,12.,16.,20.,24.,32.,40.,48.,64.,80.,96.,112.,128.)

def grid_base(h):
    yy,xx=torch.meshgrid(torch.arange(256,device=h.device,dtype=torch.float32),torch.arange(256,device=h.device,dtype=torch.float32),indexing='ij')
    return yy,xx

def basic_inputs(h,r,c):
    yy,xx=grid_base(h); b=len(h)
    dx=xx[None]-c[:,None,None];dy=yy[None]-r[:,None,None]
    ht=h[torch.arange(b,device=h.device),r,c]
    dz=h+1-(ht[:,None,None]+3)
    d=torch.sqrt(dx.square()+dy.square()+dz.square()).clamp_min(.25)
    sy,sx=torch.gradient(h,dim=(-2,-1))
    hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0]
    tx=torch.zeros_like(h);tx[torch.arange(b,device=h.device),r,c]=1
    x=torch.stack((h/100,(h-ht[:,None,None])/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(d)/6,tx,torch.exp(-(dx.square()+dy.square())/8)),1)
    freq=torch.tensor([415e6,5.775e9],device=h.device,dtype=torch.float32)
    fs=20*torch.log10(4*torch.pi*d[:,None]*freq[None,:,None,None]/299792458.)
    return x,fs

@torch.no_grad()
def geometry(h,r,c):
    """Pure geometry. Each row uses ONLY its heightmap and TX. Output fp16 normalized."""
    h=h.float();b=len(h);yy,xx=grid_base(h)
    ht=h[torch.arange(b,device=h.device),r,c]+3
    dx=xx[None]-c[:,None,None];dy=yy[None]-r[:,None,None]
    dz=h+1-ht[:,None,None]
    bins=[];ee2=torch.full_like(h,-1e5);pos=torch.zeros_like(h)
    emax=torch.full_like(h,-1e5);tmax=torch.zeros_like(h);esurf=torch.full_like(h,-1e5)
    # Midpoint samples exclude endpoints. Eight ordered intervals are preserved.
    for k in range(8):
        t=(torch.arange(k*32,(k+1)*32,device=h.device,dtype=torch.float32)+.5)/PROFILE_STEPS
        gx=(c[:,None,None,None]+dx[:,None]*t[None,:,None,None])/127.5-1
        gy=(r[:,None,None,None]+dy[:,None]*t[None,:,None,None])/127.5-1
        grid=torch.stack((gx,gy),-1).reshape(b,32*256,256,2)
        surface=F.grid_sample(h[:,None],grid,mode='bilinear',padding_mode='border',align_corners=True)[:,0].reshape(b,32,256,256)
        e=surface-(ht[:,None,None,None]+dz[:,None]*t[None,:,None,None])
        v,j=e.max(1);bins.append(v)
        better=v>emax;tmax=torch.where(better,t[j],tmax);emax=torch.maximum(emax,v)
        ee2=torch.maximum(ee2,(e+(1-t)[None,:,None,None]).amax(1))
        pos+=(e>0).sum(1)
        esurf=torch.maximum(esurf,(e+.95*t[None,:,None,None]).amax(1))
    cos=(dz-torch.gradient(h,dim=-1)[0]*dx-torch.gradient(h,dim=-2)[0]*dy)/torch.sqrt(dx.square()+dy.square()+dz.square()).clamp_min(.25)
    sy,sx=torch.gradient(h,dim=(-2,-1));cos=cos/torch.sqrt(1+sx.square()+sy.square())
    # Geometric TX-to-surface visibility proxy, not a path power or LOS label.
    txvis=torch.sigmoid(-esurf/.02)
    radius=torch.tensor(RADII,device=h.device)
    n=torch.zeros((b,3,256,256),device=h.device);den=torch.zeros_like(n)
    bestextra=torch.full_like(h,512.)
    d0=torch.sqrt(dx.square()+dy.square()+dz.square())
    for a in range(16):
        angle=2*torch.pi*a/16
        ox=radius*round(math.cos(angle),7);oy=radius*round(math.sin(angle),7)
        px=xx[None]+ox[:,None,None];py=yy[None]+oy[:,None,None]
        valid=(px>=0)&(px<=255)&(py>=0)&(py<=255)
        grid=torch.stack((px/127.5-1,py/127.5-1),-1).reshape(1,16*256,256,2).expand(b,-1,-1,-1)
        hh=F.grid_sample(h[:,None],grid,padding_mode='border',align_corners=True)[:,0].reshape(b,16,256,256)
        vv=F.grid_sample(txvis[:,None],grid,padding_mode='zeros',align_corners=True)[:,0].reshape(b,16,256,256)
        slope=(hh-(h[:,None]+1))/radius[None,:,None,None]
        horizon=torch.cummax(slope,1).values
        prior=F.pad(horizon[:,:-1],(0,0,0,0,1,0),value=-1e5)
        rxvis=(slope+.05/radius[None,:,None,None]>=prior).float()
        visible=rxvis*vv*valid[None]
        for j,(lo,hi) in enumerate(((0,5),(5,10),(10,16))):
            n[:,j]+=visible[:,lo:hi].sum(1);den[:,j]+=valid[lo:hi].sum(0)
        dt=torch.sqrt((px[None]-c[:,None,None,None]).square()+(py[None]-r[:,None,None,None]).square()+(hh+.05-ht[:,None,None,None]).square())
        dr=torch.sqrt(radius[None,:,None,None].square()+(hh+.05-h[:,None]-1).square())
        extra=(dt+dr-d0[:,None]).clamp_min(0)
        extra=torch.where(visible>.5,extra,torch.full_like(extra,512.))
        bestextra=torch.minimum(bestextra,extra.amin(1))
    g=torch.stack(bins+[ee2,pos/PROFILE_STEPS,tmax,cos],1)
    g[:,:9]=torch.asinh(g[:,:9]/2)/3
    g=torch.cat((g,n/den.clamp_min(1),torch.log1p(bestextra[:,None])/6),1)
    assert g.shape==(b,N_GEOM,256,256)
    return g.half()

def make_inputs(h,r,c,features=None):
    x,fs=basic_inputs(h,r,c)
    if features is None: features=geometry(h,r,c)
    return torch.cat((x,features.float()),1).contiguous(memory_format=torch.channels_last),fs
