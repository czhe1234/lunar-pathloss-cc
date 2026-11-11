"""C56: multibounce incident-power fields condition directional surface attention."""
import torch
from torch import nn
from torch.nn import functional as F
from src.c49_model import Model as FirstBounce
class Model(FirstBounce):
    def __init__(self,kind,state,fields,map_info,margins,radiance):
        assert kind=='multibounce'
        super().__init__('transport',state,fields,map_info,margins)
        self.kind=kind;self.node_in=nn.Linear(44,64)
        self.register_buffer('radiance',radiance,persistent=False)
        self.physical_gamma=nn.Parameter(torch.zeros(2))
        self.power_mix=nn.Parameter(torch.zeros(2,8))
    def sample_power(self,maps,qx,qy):
        side=self.radiance.shape[1];cell=256/side
        xx=((qx+.5)/cell-.5).clamp(0,side-1);yy=((qy+.5)/cell-.5).clamp(0,side-1)
        x0=xx.floor().long();y0=yy.floor().long();x1=(x0+1).clamp_max(side-1);y1=(y0+1).clamp_max(side-1)
        wx=xx-x0;wy=yy-y0;base=maps[:,None]*side*side;table=self.radiance.reshape(-1,10)
        def at(x,y):return torch.pow(10.,table[base+y*side+x].float())
        return at(x0,y0)*((1-wx)*(1-wy))[...,None]+at(x1,y0)*(wx*(1-wy))[...,None]+at(x0,y1)*((1-wx)*wy)[...,None]+at(x1,y1)*(wx*wy)[...,None]
    def forward(self,x,maps):
        z=self.global_blocks(F.silu(self.global_in(self.embedding(x))))
        features,prior,inside=super().nodes(x,maps)
        info=self.map_info[maps];rx=info[:,2].float()+x[:,5]*128;ry=info[:,1].float()+x[:,6]*128
        qx=rx[:,None]+features[:,:,22]*128;qy=ry[:,None]+features[:,:,23]*128
        power=self.sample_power(maps,qx,qy);B=power[:,:,:8];E=power[:,:,8:]
        direct=E[:,:,:,None].expand(-1,-1,-1,4).reshape(len(x),-1,8)
        rho=torch.tensor([.2,.5,.8,.95]*2,device=x.device)
        extra=torch.cat((B.clamp_min(1e-30).log10()/8,
          (B.clamp_min(1e-30).log10()-(direct*rho).clamp_min(1e-30).log10()).clamp(-1,30)/8,
          E.clamp_min(1e-30).log10()/8),-1)
        nodes=self.node_blocks(F.silu(self.node_in(torch.cat((features,extra),-1))))
        q=self.query(z).reshape(len(x),2,64);keys=self.key(nodes);values=self.value(nodes)
        with torch.autocast('cuda',enabled=False):
            norm=torch.sqrt(1+25*(features[:,:,2].square()+features[:,:,3].square()))
            dr=torch.expm1(features[:,:,8]*6).clamp_min(.25)
            frac=torch.remainder(x[:,31],1)
            d0=(x[:,5]*128).square()+(x[:,6]*128).square()+(x[:,7]*50+frac).square()
            area=self.area[None,None].expand(len(x),16,-1).reshape(len(x),-1)
            receive=4*area*norm*features[:,:,10]*d0[:,None]/dr.square()*features[:,:,17]*inside
            illumination=torch.einsum('nle,be->nbl',B.float(),self.power_mix.float().softmax(-1))
            multi=illumination*receive[:,None]
            boost=torch.log1p(multi/(prior[:,None]+1e-20)).clamp_max(40)
            scores=torch.einsum('nbc,nlc->nbl',q.float(),keys.float())/8
            scores=scores+torch.log((prior[:,None]+self.visibility_floor.exp()[None,:,None]).clamp_min(1e-20))+self.physical_gamma[None,:,None]*boost
            scores=scores.masked_fill(~inside[:,None],-1e4)
            context=torch.einsum('nbl,nlc->nbc',scores.softmax(-1),values.float())
        gate=self.gate(z).reshape(len(x),2,4).float().softmax(-1);out=[]
        for b,head in enumerate(self.decoders):
            hidden=head[:-1](torch.cat((z,context[:,b]),-1))
            with torch.autocast('cuda',enabled=False):delta=head[-1](hidden.float())
            out.append((delta*gate[:,b]).sum(-1))
        return torch.stack(out,-1)
