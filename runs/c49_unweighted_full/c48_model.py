"""C48: actual2D raw terrain around propagation corridors and RX, plus294 descriptors."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.c34_model import NumericEmbedding
from src.c47_model import DenseBlock
class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+.2*self.net(x)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(19,24,3,padding=1),Block(24),
          nn.Conv2d(24,48,3,stride=2,padding=1),Block(48),nn.Conv2d(48,64,3,stride=(2,1),padding=1),Block(64))
    def forward(self,x):return self.net(x.contiguous(memory_format=torch.channels_last)).flatten(2).transpose(1,2)
class Model(nn.Module):
    def __init__(self,kind,state,fields,map_info):
        super().__init__();assert kind in ['strip','patch','both'];self.kind=kind
        self.register_buffer('fields',fields,persistent=False)
        self.register_buffer('map_info',map_info,persistent=False)
        self.embedding=NumericEmbedding('periodic',state['embedding.mean'],state['embedding.std'],state['embedding.edges'])
        self.embedding.load_state_dict({k[len('embedding.'):]:v for k,v in state.items() if k.startswith('embedding.')})
        self.global_in=nn.Linear(294*9,384);self.global_blocks=nn.Sequential(*[DenseBlock(384) for _ in range(3)])
        if kind in ['strip','both']:self.strip=Encoder()
        if kind in ['patch','both']:self.patch=Encoder()
        self.input_type=nn.Parameter(torch.randn(2,64)*.02)
        self.query=nn.Linear(384,128);self.key=nn.Linear(64,64);self.value=nn.Linear(64,64)
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(448,256),nn.SiLU(),nn.Linear(256,4)) for _ in range(2)])
        self.gate=nn.Linear(384,8)
        for h in self.decoders:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
        self.register_buffer('t',(torch.arange(128,dtype=torch.float32)+.5)/128,persistent=False)
        self.register_buffer('cross',torch.tensor([-4,-3,-2,-1.5,-1,-.6,-.3,-.15,0,.15,.3,.6,1,1.5,2,3,4]),persistent=False)
        self.register_buffer('near',torch.tensor([-64,-32,-16,-8,-4,-2,-1,-.5,0,.5,1,2,4,8,16,32,64]),persistent=False)
    def sample(self,hm,qx,qy):
        shape=qx.shape;xx=qx.clamp(0,255);yy=qy.clamp(0,255)
        x0=xx.floor().long();y0=yy.floor().long();x1=(x0+1).clamp_max(255);y1=(y0+1).clamp_max(255)
        wx=xx-x0;wy=yy-y0;field=self.fields.reshape(-1,3)
        base=hm[:,None,None]*65536
        def at(x,y):return field[(base+y*256+x).reshape(-1)].reshape(*shape,3)
        return at(x0,y0)*((1-wx)*(1-wy))[...,None]+at(x1,y0)*(wx*(1-wy))[...,None]+at(x0,y1)*((1-wx)*wy)[...,None]+at(x1,y1)*(wx*wy)[...,None]
    def image(self,x,maps,kind):
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            x=x.float();info=self.map_info[maps];hm=info[:,0].long();tr=info[:,1].float();tc=info[:,2].float()
            dx=x[:,5]*128;dy=x[:,6]*128;dh=torch.sqrt(dx.square()+dy.square()).clamp_min(.25);ux=dx/dh;uy=dy/dh
            rx=tc+dx;ry=tr+dy;txz=x[:,31]+3;rz=x[:,0]*100+1;dz=rz-txz
            if kind=='strip':
                t=self.t[None,:,None];radius=torch.sqrt((299792458/415e6)*dh[:,None,None]*t*(1-t)).clamp_min(.25)
                off=radius*self.cross[None,None]
                qx=tc[:,None,None]+dx[:,None,None]*t-uy[:,None,None]*off
                qy=tr[:,None,None]+dy[:,None,None]*t+ux[:,None,None]*off
            else:
                along=self.near[None,:,None];cross=self.near[None,None,:]
                qx=rx[:,None,None]+ux[:,None,None]*along-uy[:,None,None]*cross
                qy=ry[:,None,None]+uy[:,None,None]*along+ux[:,None,None]*cross
            valid=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255)
            sampled=self.sample(hm,qx,qy);z,sx,sy=sampled.unbind(-1)
            t=((qx-tc[:,None,None])*ux[:,None,None]+(qy-tr[:,None,None])*uy[:,None,None])/dh[:,None,None]
            e=z-(txz[:,None,None]+t*dz[:,None,None]);frac=torch.remainder(x[:,31],1)
            alt=e+frac[:,None,None]*(1-t)
            dt=torch.sqrt((qx-tc[:,None,None]).square()+(qy-tr[:,None,None]).square()+(z-txz[:,None,None]).square()).clamp_min(.25)
            dr=torch.sqrt((qx-rx[:,None,None]).square()+(qy-ry[:,None,None]).square()+(z-rz[:,None,None]).square()).clamp_min(.25)
            direct=torch.sqrt(dh.square()+dz.square());ex=(dt+dr-direct[:,None,None]).clamp_min(0)
            vf=torch.sqrt(2*(dt+dr)/(dt*dr))
            phase0=2*math.pi*ex/(299792458/415e6);phase1=2*math.pi*ex/(299792458/5.775e9)
            values=[torch.asinh(e/2)/3,torch.asinh(alt/2)/3,
                (qx-tc[:,None,None])/128,(qy-tr[:,None,None])/128,
                (qx-rx[:,None,None])/128,(qy-ry[:,None,None])/128,
                torch.log1p(dt)/6,torch.log1p(dr)/6,
                torch.asinh(e*vf/math.sqrt(299792458/415e6))/3,torch.asinh(e*vf/math.sqrt(299792458/5.775e9))/3,
                phase0.sin(),phase0.cos(),phase1.sin(),phase1.cos(),
                (sx*ux[:,None,None]+sy*uy[:,None,None])/5,(-sx*uy[:,None,None]+sy*ux[:,None,None])/5,
                t.clamp(-2,3),torch.log1p(ex)/4]
            out=torch.stack(values,1)*valid[:,None];return torch.cat((out,valid[:,None].float()),1)
    def forward(self,x,maps):
        z=self.global_blocks(F.silu(self.global_in(self.embedding(x))))
        tokens=[]
        if self.kind in ['strip','both']:tokens.append(self.strip(self.image(x,maps,'strip'))+self.input_type[0])
        if self.kind in ['patch','both']:tokens.append(self.patch(self.image(x,maps,'patch'))+self.input_type[1])
        tokens=torch.cat(tokens,1);q=self.query(z).reshape(len(x),2,64);keys=self.key(tokens);values=self.value(tokens)
        with torch.autocast('cuda',enabled=False):
            w=torch.einsum('nbc,nlc->nbl',q.float(),keys.float())/8
            context=torch.einsum('nbl,nlc->nbc',w.softmax(-1),values.float())
        gates=self.gate(z).reshape(len(x),2,4).float().softmax(-1);pred=[]
        for b,h in enumerate(self.decoders):
            hidden=h[:-1](torch.cat((z,context[:,b]),-1))
            with torch.autocast('cuda',enabled=False):d=h[-1](hidden.float())
            pred.append((d*gates[:,b]).sum(-1))
        return torch.stack(pred,-1)
