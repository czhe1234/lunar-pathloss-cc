"""Coarse diffuse surface-power transport from H/TX only; no simulation labels."""
import math
import torch
from torch.nn import functional as F
class Geometry:
    def __init__(self,side=32,neighbors=128,samples=128):
        self.side=side;self.samples=samples
        a=(torch.arange(side,device='cuda',dtype=torch.float32)+.5)*256/side-.5
        y,x=torch.meshgrid(a,a,indexing='ij');self.xy=torch.stack((x,y),-1).reshape(-1,2);self.N=len(self.xy)
        d=torch.cdist(self.xy,self.xy);d.fill_diagonal_(float('inf'))
        self.neighbors=d.topk(neighbors,largest=False,sorted=False).indices
        self.grid=(self.xy/127.5-1)[None,:,None];self.area=(256/side)**2
    @torch.inference_mode()
    def build(self,h):
        B=len(h);sy,sx=torch.gradient(h,dim=(-2,-1));fields=torch.stack((h,sx,sy),1)
        val=F.grid_sample(fields,self.grid.expand(B,-1,-1,-1),align_corners=True,padding_mode='border')[:,:,:,0].transpose(1,2)
        z=val[:,:,0];normal=torch.stack((-val[:,:,1],-val[:,:,2],torch.ones_like(z)),-1);norm=normal.norm(dim=-1);normal/=norm[...,None]
        pos=torch.cat((self.xy[None].expand(B,-1,-1),(z+.05)[...,None]),-1)
        vec=pos[:,self.neighbors]-pos[:,:,None];distance=vec.norm(dim=-1).clamp_min(.1);unit=vec/distance[...,None]
        cr=(normal[:,:,None]*unit).sum(-1).clamp_min(0)
        ce=-(normal[:,self.neighbors]*unit).sum(-1);ce=ce.clamp_min(0)
        margin=torch.full(distance.shape,-1e6,device='cuda')
        for lo in range(0,self.samples,8):
            t=(torch.arange(lo,min(lo+8,self.samples),device='cuda',dtype=torch.float32)+.5)/self.samples
            q=pos[:,:,None,None]+vec[:,:,:,None]*t[None,None,None,:,None]
            grid=(q[...,:2]/127.5-1).reshape(B,-1,len(t),2)
            height=F.grid_sample(h[:,None],grid,align_corners=True,padding_mode='border')[:,0].reshape(B,self.N,-1,len(t))
            margin=torch.maximum(margin,(height-q[...,2]).amax(-1))
        values=self.area*norm[:,self.neighbors]*cr*ce*(margin<=0)/(math.pi*distance.square())
        raw=values.sum(-1);values*=torch.clamp(.95/raw.clamp_min(1e-9),max=1)[:,:,None]
        K=torch.zeros((B,self.N,self.N),device='cuda')
        K.scatter_(2,self.neighbors[None].expand(B,-1,-1),values)
        return K,pos,normal,raw
    @torch.inference_mode()
    def radiance(self,h,K,pos,normal,txr,txc,margins):
        """Returns log10 exitance for TX heights3/candidate and rho.2/.5/.8/.95, plus direct irradiance."""
        B,M=txr.shape;N=self.N
        em=F.grid_sample(margins.reshape(B*M,256,256,2).permute(0,3,1,2).float(),
          self.grid.expand(B*M,-1,-1,-1),align_corners=True,padding_mode='border')[:,:,:,0].reshape(B,M,2,N)
        bi=torch.arange(B,device='cuda')[:,None];ht=h[bi,txr,txc];frac=torch.remainder(ht,1)
        ztx=torch.stack((ht+3,ht+3-frac),2)
        tx=torch.stack((txc.float(),txr.float()),-1)[:,:,None,None].expand(-1,-1,2,N,-1)
        tx=torch.cat((tx,ztx[:,:,:,None,None].expand(-1,-1,-1,N,1)),-1)
        v=tx-pos[:,None,None];dist=v.norm(dim=-1).clamp_min(.25);ci=(normal[:,None,None]*v/dist[...,None]).sum(-1).clamp_min(0)
        ev=torch.stack((em[:,:,0],(1-frac[:,:,None])*em[:,:,0]+frac[:,:,None]*em[:,:,1]),2)
        visible=torch.sigmoid(-ev/.03);E=visible*ci/(4*math.pi*dist.square())
        rho=torch.tensor([.2,.5,.8,.95],device='cuda')[None,None,None,:,None]
        source=E[:,:,:,None].expand(-1,-1,-1,4,-1).reshape(B,M*8,N)
        reflect=rho.expand(B,M,2,4,N).reshape(B,M*8,N)
        exitance=reflect*source;relative=float('inf');iterations=0
        for it in range(1,65):
            new=reflect*(source+torch.bmm(exitance,K.transpose(1,2)))
            if it%8==0:
                relative=float(((new-exitance).abs()/(new.abs()+1e-30)).max())
            exitance=new;iterations=it
            if it%8==0 and relative<1e-6:break
        assert torch.isfinite(exitance).all() and (exitance>=0).all()
        assert (exitance+1e-25>=reflect*source*(1-1e-5)).all()
        values=exitance.reshape(B,M,2,4,N).permute(0,1,4,2,3).reshape(B,M,N,8)
        direct=E.permute(0,1,3,2)
        out=torch.cat((values,direct),-1).clamp_min(1e-30).log10()
        return out,{'iterations':iterations,'maximum_last_relative_update':relative}
