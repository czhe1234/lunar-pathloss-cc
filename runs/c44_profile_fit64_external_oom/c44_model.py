"""C44: C43 plus a learned ordered128-point profile branch.

B (grid) and C (path) have identical state dictionaries and initialization.
Only reference sampling positions differ. No label/mask/base-model access.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c),
                               nn.SiLU(),nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x):return F.silu(x+self.net(x))

def project(a,b):return nn.Sequential(nn.Conv2d(a,b,1,bias=False),nn.GroupNorm(8,b))

class TerrainEncoder(nn.Module):
    def __init__(self):
        super().__init__();cs=[32,64,128,192];self.cs=cs
        self.stem=nn.Sequential(nn.Conv2d(11,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.SiLU(),Residual(32))
        self.down=nn.ModuleList([nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU(),Residual(b)) for a,b in zip(cs,cs[1:])])
        self.blocks=nn.ModuleList([nn.ModuleList([Residual(c) for c in cs]) for _ in range(3)])
        self.up=nn.ModuleList([nn.ModuleList([project(cs[j+1],cs[j]) for j in range(3)]) for _ in range(3)])
        self.dn=nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Conv2d(cs[j],cs[j+1],3,stride=2,padding=1,bias=False),nn.GroupNorm(8,cs[j+1])) for j in range(3)]) for _ in range(3)])
        self.local_proj=nn.ModuleList([nn.Conv2d(c,24,1) for c in cs])
        self.local=nn.Sequential(nn.Conv2d(96+11,96,1),nn.SiLU(),Residual(96))
    def forward(self,x):
        h=[self.stem(x)]
        for down in self.down:h.append(down(h[-1]))
        for stage in range(3):
            new=[]
            for j in range(4):
                v=h[j]
                if j>0:v=v+self.dn[stage][j-1](h[j-1])
                if j<3:v=v+F.interpolate(self.up[stage][j](h[j+1]),size=h[j].shape[-2:],mode='bilinear',align_corners=True)
                new.append(self.blocks[stage][j](v))
            h=new
        local=self.local(torch.cat([x]+[F.interpolate(p(v),size=x.shape[-2:],mode='bilinear',align_corners=True) for p,v in zip(self.local_proj,h)],1))
        return h,local

def at_pixels(a,pixels):
    return a.flatten(2).gather(2,pixels[:,None].expand(-1,a.shape[1],-1)).transpose(1,2)

class OrderedProfile(nn.Module):
    """128 ordered clearances; raster-height phase is an empirical proxy."""
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(4,16,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(16,16,3,padding=d,dilation=d),nn.SiLU(),nn.Conv1d(16,16,1)) for d in [1,2,4,8,16,32]])
        self.out=nn.Linear(32,96)
        nn.init.normal_(self.out.weight,std=.005);nn.init.zeros_(self.out.bias)
        self.register_buffer('t',(torch.arange(128,dtype=torch.float32)+.5)/128,persistent=False)
    def forward(self,raw,tx,rx,rz):
        B,Q,_=rx.shape;t=self.t
        with torch.autocast('cuda',enabled=False):
            xy=tx[:,None,None,:2]+(rx[:,:,None]-tx[:,None,None,:2])*t[None,None,:,None]
            height=F.grid_sample(raw[:,:1].float()*100,xy/127.5-1,align_corners=True,padding_mode='border')[:,0]
            e=height-(tx[:,None,None,2]+(rz-tx[:,None,2])[:,:,None]*t)
            # This candidate is not asserted to be the actual continuous TX.
            frac=torch.remainder(tx[:,2]-3,1)[:,None,None]
            candidate=e+frac*(1-t)
            distance=torch.sqrt((rx-tx[:,None,:2]).square().sum(-1)+(rz-tx[:,None,2]).square()).clamp_min(.25)
            seq=torch.stack((torch.asinh(e/2)/3,torch.asinh(candidate/2)/3,t[None,None].expand(B,Q,-1),torch.log1p(distance)[:,:,None].expand(B,Q,128)/6),-1)
        z=F.silu(self.stem(seq.reshape(B*Q,128,4).transpose(1,2)))
        for block in self.blocks:z=F.silu(z+.1*block(z))
        z=torch.cat((z.mean(-1),z.amax(-1)),1)
        return self.out(z).reshape(B,Q,96)

class Model(nn.Module):
    def __init__(self,variant='path'):
        super().__init__();assert variant in ['cnn','grid','path']
        self.variant=variant;self.width=64;self.heads=4;self.points=16
        self.encoder=TerrainEncoder()
        self.query=nn.Sequential(nn.Linear(96+11+32,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64))
        self.kv=nn.ModuleList([nn.Conv2d(c,128,1) for c in self.encoder.cs])
        self.offset=nn.Linear(64,32);nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        self.geom=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,64))
        self.scale_embedding=nn.Parameter(torch.randn(4,64)*.02)
        self.band=nn.Parameter(torch.randn(2,64)*.02)
        self.attout=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.LayerNorm(64))
        self.readout=nn.ModuleList([nn.Sequential(nn.Linear(96+11+32+64,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1)) for _ in range(2)])
        for head in self.readout:
            nn.init.normal_(head[-1].weight,std=.005);nn.init.zeros_(head[-1].bias)
        yy,xx=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
        self.register_buffer('xy',torch.stack((xx,yy)).flatten(1).T,persistent=False)
        self.register_buffer('freqs',2*math.pi*2**torch.arange(4,dtype=torch.float32),persistent=False)
        gy,gx=torch.meshgrid(torch.tensor([32.,96.,160.,224.]),torch.tensor([32.,96.,160.,224.]),indexing='ij')
        self.register_buffer('grid_refs',torch.stack((gx,gy),-1).reshape(16,2),persistent=False)
        angle=torch.arange(8)*2*math.pi/8
        radius=torch.tensor([8.,16.,32.,64.,96.,128.,64.,32.])
        self.register_buffer('ring',torch.stack((angle.cos()*radius,angle.sin()*radius),-1),persistent=False)
        self.profile=OrderedProfile()
    def encode(self,x,tx):
        fields,local=self.encoder(x)
        return {'local':local,'raw':x,'tx':tx,'kv':[p(f) for p,f in zip(self.kv,fields)]}
    def references(self,rx,tx,q):
        if self.variant=='path':
            t=torch.linspace(.05,.95,8,device=rx.device)
            line=tx[:,None,None,:2]+(rx[:,:,None]-tx[:,None,None,:2])*t[None,None,:,None]
            ring=rx[:,:,None]+self.ring[None,None]
            base=torch.cat((line,ring),2)
        else:base=self.grid_refs[None,None].expand(rx.shape[0],rx.shape[1],-1,-1)
        return (base+32*torch.tanh(self.offset(q).float()).reshape(*q.shape[:2],16,2)).clamp(0,255)
    def decode(self,state,pixels):
        local=at_pixels(state['local'],pixels);raw=at_pixels(state['raw'],pixels)
        local=local+self.profile(state['raw'],state['tx'],self.xy[pixels],raw[...,0].float()*100+1)
        rx=self.xy[pixels];tx=state['tx']
        coords=torch.cat((rx/128-1,(rx-tx[:,None,:2])/128),-1)
        angle=coords[:,:,:,None]*self.freqs
        pos=torch.cat((angle.sin(),angle.cos()),-1).flatten(2)
        inp=torch.cat((local,raw,pos),-1);q=self.query(inp)
        if self.variant=='cnn':contexts=[torch.zeros_like(q),torch.zeros_like(q)]
        else:
            refs=self.references(rx,tx,q)
            with torch.autocast('cuda',enabled=False):
                grid=refs.float()/127.5-1
                # Sample the same raw surface used to define the query.
                geo=F.grid_sample(state['raw'][:,[0,3,4]].float(),grid,align_corners=True,padding_mode='border').permute(0,2,3,1)
                z=geo[...,0]*100;sx=geo[...,1]*5;sy=geo[...,2]*5
                rz=raw[...,0].float()*100+1
                vrx=torch.cat((refs-rx[:,:,None],(z-rz[:,:,None])[...,None]),-1)
                vtx=torch.cat((refs-tx[:,None,None,:2],(z-tx[:,None,None,2])[...,None]),-1)
                dr=vrx.square().sum(-1).clamp_min(.0625).sqrt();dt=vtx.square().sum(-1).clamp_min(.0625).sqrt()
                direct=torch.sqrt((rx-tx[:,None,:2]).square().sum(-1)+(rz-tx[:,None,2]).square()).clamp_min(.25)
                ray=rx-tx[:,None,:2]
                t=((refs-tx[:,None,None,:2])*ray[:,:,None]).sum(-1)/ray.square().sum(-1).clamp_min(1)[:,:,None]
                clearance=z-(tx[:,None,None,2]+t*(rz-tx[:,None,2])[:,:,None])
                norm=torch.sqrt(1+sx*sx+sy*sy)
                ni=(-sx*(-vtx[...,0])-sy*(-vtx[...,1])-vtx[...,2])/norm/dt
                no=(-sx*(-vrx[...,0])-sy*(-vrx[...,1])-vrx[...,2])/norm/dr
                geom=torch.stack((vrx[...,0]/128,vrx[...,1]/128,vrx[...,2]/20,
                                  torch.log1p(dt)/6,torch.log1p(dr)/6,
                                  torch.log1p((dt+dr-direct[:,:,None]).clamp_min(0))/6,
                                  t.clamp(-2,3),torch.asinh(clearance/2)/3,ni,no),-1)
                samples=[F.grid_sample(a.float(),grid,align_corners=True,padding_mode='border').permute(0,2,3,1) for a in state['kv']]
            gp=self.geom(geom)
            keys=torch.cat([v[...,:64]+gp+self.scale_embedding[i] for i,v in enumerate(samples)],2)
            vals=torch.cat([v[...,64:] for v in samples],2)
            B,Q,K,_=keys.shape
            kk=keys.reshape(B,Q,K,4,16);vv=vals.reshape(B,Q,K,4,16)
            contexts=[]
            for b in range(2):
                qq=(q+self.band[b]).reshape(B,Q,4,16)
                with torch.autocast('cuda',enabled=False):
                    scores=torch.einsum('bqhd,bqkhd->bqhk',qq.float(),kk.float())/4
                    weights=scores.softmax(-1)
                    context=torch.einsum('bqhk,bqkhd->bqhd',weights,vv.float()).flatten(2)
                contexts.append(self.attout(context))
        preds=[]
        for b in range(2):
            hidden=self.readout[b][:-1](torch.cat((inp,contexts[b]),-1))
            with torch.autocast('cuda',enabled=False):preds.append(self.readout[b][-1](hidden.float()).squeeze(-1))
        return torch.stack(preds,-1)
    def forward(self,x,tx,pixels):return self.decode(self.encode(x,tx),pixels)
