"""Independent straight-path Jones transmission through candidate terrain facets."""
import math
import torch
def slab(cosine,freq,sigma,eps=3.411,thickness=10.):
    c=cosine.float().clamp(1e-4,1)
    eta=torch.complex(torch.full_like(freq,eps),-sigma/(2*math.pi*freq*8.8541878128e-12))
    root=torch.sqrt(eta-(1-c[...,None].square()))
    rS=(c[...,None]-root)/(c[...,None]+root)
    rP=(eta*c[...,None]-root)/(eta*c[...,None]+root)
    phase=2*math.pi*thickness*freq/299792458*root
    q=torch.exp(-1j*phase);q2=q*q
    TS=(1-rS*rS)*q/(1-rS*rS*q2);TP=(1-rP*rP)*q/(1-rP*rP*q2)
    RS=rS*(1-q2)/(1-rS*rS*q2);RP=rP*(1-q2)/(1-rP*rP*q2)
    return TS,TP,RS,RP
@torch.inference_mode()
def transmission(x,hm,Hflat,max_crossings=32):
    x=x.float();dev=x.device;N=len(x)
    txr=x[:,32]*256;txc=x[:,33]*256;dx=x[:,5]*128;dy=x[:,6]*128
    freq=torch.tensor([415e6,5.775e9],device=dev);sigma=torch.tensor([.0003562,.009324],device=dev)
    t=torch.cat((torch.zeros(1,device=dev),(torch.arange(128,device=dev)+.5)/128,torch.ones(1,device=dev)))
    nominal=2*torch.sinh(3*x[:,104:232]);fraction=torch.remainder(x[:,31],1)
    results=[];counts=[];grazing=[]
    for phase in [False,True]:
        frac=fraction if phase else torch.zeros_like(fraction)
        e=torch.cat((-(3-frac)[:,None],nominal+frac[:,None]*(1-t[1:-1]),-torch.ones((N,1),device=dev)),1)
        left,right=e[:,:-1],e[:,1:];change=(left>0)!=(right>0)
        delta=right-left;delta=torch.where(delta.abs()>1e-8,delta,torch.ones_like(delta))
        roots=t[:-1]+(-left/delta)*(t[1:]-t[:-1])
        roots=torch.where(change,roots,torch.full_like(roots,2.))
        selected=roots.topk(max_crossings,dim=1,largest=False,sorted=True).values;valid=selected<=1
        qx=txc[:,None]+dx[:,None]*selected;qy=txr[:,None]+dy[:,None]*selected
        xx=qx.clamp(0,255);yy=qy.clamp(0,255);x0=xx.floor().long().clamp_max(254);y0=yy.floor().long().clamp_max(254)
        fx=xx-x0;fy=yy-y0;offset=hm[:,None]*65536+y0*256+x0
        h00=Hflat[offset];h10=Hflat[offset+1];h01=Hflat[offset+256];h11=Hflat[offset+257]
        sxB=(1-fy)*(h10-h00)+fy*(h11-h01);syB=(1-fx)*(h01-h00)+fx*(h11-h10)
        upper=fy<=fx;sx0=torch.where(upper,h10-h00,h11-h01);sy0=torch.where(upper,h11-h10,h01-h00)
        lower=fx+fy<=1;sx1=torch.where(lower,h10-h00,h11-h01);sy1=torch.where(lower,h01-h00,h11-h10)
        sx=torch.stack((sxB,sx0,sx1),1);sy=torch.stack((syB,sy0,sy1),1)
        normal=torch.stack((-sx,-sy,torch.ones_like(sx)),-1);normal/=normal.norm(dim=-1,keepdim=True)
        dz=x[:,7]*50+frac;k=torch.stack((dx,dy,dz),1);k/=k.norm(dim=1,keepdim=True).clamp_min(1e-6)
        horizontal=torch.stack((-k[:,1],k[:,0],torch.zeros(N,device=dev)),1)
        fallback=torch.zeros_like(horizontal);fallback[:,1]=1
        horizontal=torch.where((horizontal.norm(dim=1)<1e-6)[:,None],fallback,horizontal)
        horizontal/=horizontal.norm(dim=1,keepdim=True)
        vertical=torch.linalg.cross(horizontal,k,dim=-1)
        kk=k[:,None,None].expand_as(normal);s=torch.linalg.cross(kk,normal,dim=-1)
        sn=s.norm(dim=-1,keepdim=True);s=torch.where(sn<1e-6,horizontal[:,None,None],s/sn.clamp_min(1e-8))
        p=torch.linalg.cross(s,kk,dim=-1);cosine=(kk*normal).sum(-1).abs().clamp(1e-4,1)
        TS,TP,_,_=slab(cosine,freq,sigma)
        field=vertical[:,None,None].expand(-1,3,2,-1).to(torch.complex64)
        for j in range(max_crossings):
            sj=s[:,:,j,None];pj=p[:,:,j,None]
            new=(TS[:,:,j]*(field*sj).sum(-1))[...,None]*sj+(TP[:,:,j]*(field*pj).sum(-1))[...,None]*pj
            field=torch.where(valid[:,j,None,None,None],new,field)
        loss=-10*field.abs().square().sum(-1).clamp_min(1e-30).log10()
        assert torch.isfinite(loss).all()
        results.append(loss);counts.append(change.sum(1))
        g=(-torch.log10(cosine.clamp_min(1e-4))*valid[:,None]).sum(2);grazing.append(g)
    return torch.stack(results,1),torch.stack(counts,1),torch.stack(grazing,1)
