"""Exact straight-ray intersection for candidate1m triangular terrain meshes."""
import torch
from src.c57_transmission import slab
def terrain(x,y,hm,H,diagonal):
    x=x.clamp(0,255);y=y.clamp(0,255);ix=x.floor().long().clamp_max(254);iy=y.floor().long().clamp_max(254)
    fx=x-ix;fy=y-iy;i=hm[:,None]*65536+iy*256+ix
    a=H[i];b=H[i+1];c=H[i+256];d=H[i+257]
    if diagonal==0:
        upper=fy<=fx;sx=torch.where(upper,b-a,d-c);sy=torch.where(upper,d-b,c-a)
        height=a+sx*fx+sy*fy
    else:
        lower=fx+fy<=1;sx=torch.where(lower,b-a,d-c);sy=torch.where(lower,c-a,d-b)
        height=torch.where(lower,a+sx*fx+sy*fy,d+sx*(fx-1)+sy*(fy-1))
    return height,sx,sy
@torch.inference_mode()
def trace(x,hm,H,phase=False,diagonal=0,rx_offset=(0.,0.),tx_offset=(0.,0.),horizontal_scale=1.):
    N=len(x);dev=x.device
    txr=x[:,32]*256+tx_offset[1];txc=x[:,33]*256+tx_offset[0]
    rxr=x[:,32]*256+x[:,6]*128+rx_offset[1];rxc=x[:,33]*256+x[:,5]*128+rx_offset[0]
    rxh,_,_=terrain(rxc[:,None],rxr[:,None],hm,H,diagonal)
    txz=x[:,31]+3-(torch.remainder(x[:,31],1) if phase else 0)
    dx=rxc-txc;dy=rxr-txr;dz=rxh[:,0]+1-txz
    lines=torch.arange(256,device=dev,dtype=torch.float32)[None]
    def cuts(coord,direction):
        safe=torch.where(direction.abs()>1e-8,direction,torch.ones_like(direction))
        t=(lines-coord[:,None])/safe[:,None]
        return torch.where((direction.abs()>1e-8)[:,None]&(t>0)&(t<1),t,torch.ones_like(t))
    t=torch.cat((torch.zeros((N,1),device=dev),cuts(txc,dx),cuts(txr,dy),torch.ones((N,1),device=dev)),1).sort(1).values
    a,b=t[:,:-1],t[:,1:];mid=(a+b)/2
    mx=txc[:,None]+dx[:,None]*mid;my=txr[:,None]+dy[:,None]*mid
    ix=mx.floor().clamp(0,254);iy=my.floor().clamp(0,254)
    denominator=dx-dy if diagonal==0 else dx+dy
    numerator=ix-iy-txc[:,None]+txr[:,None] if diagonal==0 else ix+iy+1-txc[:,None]-txr[:,None]
    safe=torch.where(denominator.abs()>1e-8,denominator,torch.ones_like(denominator))
    split=numerator/safe[:,None]
    split=torch.where((denominator.abs()>1e-8)[:,None],split,a).maximum(a).minimum(b)
    ta=torch.stack((a,split),-1).flatten(1);tb=torch.stack((split,b),-1).flatten(1);tm=(ta+tb)/2
    qx=txc[:,None]+dx[:,None]*tm;qy=txr[:,None]+dy[:,None]*tm
    height,sx,sy=terrain(qx,qy,hm,H,diagonal)
    e=height-(txz[:,None]+dz[:,None]*tm);de=sx*dx[:,None]+sy*dy[:,None]-dz[:,None]
    safe=torch.where(de.abs()>1e-9,de,torch.ones_like(de))
    root=tm-e/safe
    valid=(tb-ta>1e-8)&(de.abs()>1e-9)&(root>=ta-1e-7)&(root<tb-1e-7)&(root>0)&(root<1)
    root=torch.where(valid,root,torch.full_like(root,2))
    roots,indices=root.topk(32,dim=1,largest=False,sorted=True);keep=roots<1
    sx=sx.gather(1,indices)/horizontal_scale;sy=sy.gather(1,indices)/horizontal_scale
    normal=torch.stack((-sx,-sy,torch.ones_like(sx)),-1);normal/=normal.norm(dim=-1,keepdim=True)
    k=torch.stack((dx*horizontal_scale,dy*horizontal_scale,dz),1);k/=k.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return roots,normal,keep,k,valid.sum(1)
@torch.inference_mode()
def evaluate(x,hm,H,rx_offset=(0.,0.),tx_offset=(0.,0.),horizontal_scale=1.):
    dev=x.device;N=len(x);freq=torch.tensor([415e6,5.775e9],device=dev);sigma=torch.tensor([.0003562,.009324],device=dev)
    results=[];counts=[]
    for phase in [False,True]:
        methods=[];cc=[]
        for diagonal in [0,1]:
            roots,n,valid,k,count=trace(x,hm,H,phase,diagonal,rx_offset,tx_offset,horizontal_scale)
            h=torch.stack((-k[:,1],k[:,0],torch.zeros(N,device=dev)),1);fallback=torch.zeros_like(h);fallback[:,1]=1
            h=torch.where((h.norm(dim=1)<1e-6)[:,None],fallback,h);h/=h.norm(dim=1,keepdim=True)
            v=torch.linalg.cross(h,k,dim=-1)
            kk=k[:,None].expand_as(n);s=torch.linalg.cross(kk,n,dim=-1);sn=s.norm(dim=-1,keepdim=True)
            s=torch.where(sn<1e-6,h[:,None],s/sn.clamp_min(1e-8));p=torch.linalg.cross(s,kk,dim=-1)
            cosine=(kk*n).sum(-1).abs().clamp(1e-4,1)
            ts,tp,_,_=slab(cosine,freq,sigma)
            field=v[:,None].expand(-1,2,-1).to(torch.complex64)
            for j in range(32):
                sj=s[:,j,None];pj=p[:,j,None]
                new=(ts[:,j]*(field*sj).sum(-1))[...,None]*sj+(tp[:,j]*(field*pj).sum(-1))[...,None]*pj
                field=torch.where(valid[:,j,None,None],new,field)
            methods.append(-10*field.abs().square().sum(-1).clamp_min(1e-30).log10());cc.append(count)
        results.append(torch.stack(methods,1));counts.append(torch.stack(cc,1))
    return torch.stack(results,1),torch.stack(counts,1)
