"""C49: directional surface-power quadrature with input-only visibility."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.c34_model import NumericEmbedding
from src.c47_model import DenseBlock
from src.c48_model import Model as Sampler
class Model(nn.Module):
    sample=Sampler.sample
    def __init__(self,kind,state,fields,map_info,margins):
        super().__init__();assert kind in ['transport','unweighted'];self.kind=kind
        for name,value in [('fields',fields),('map_info',map_info),('margins',margins)]:self.register_buffer(name,value,persistent=False)
        self.embedding=NumericEmbedding('periodic',state['embedding.mean'],state['embedding.std'],state['embedding.edges'])
        self.embedding.load_state_dict({k[len('embedding.'):]:v for k,v in state.items() if k.startswith('embedding.')})
        self.global_in=nn.Linear(294*9,384);self.global_blocks=nn.Sequential(*[DenseBlock(384) for _ in range(3)])
        self.node_in=nn.Linear(26,64);self.node_blocks=nn.Sequential(DenseBlock(64),DenseBlock(64))
        self.query=nn.Linear(384,128);self.key=nn.Linear(64,64);self.value=nn.Linear(64,64)
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(448,256),nn.SiLU(),nn.Linear(256,4)) for _ in range(2)])
        self.gate=nn.Linear(384,8);self.visibility_floor=nn.Parameter(torch.full((2,),-12.))
        for h in self.decoders:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
        r=torch.tensor([1.,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,72,80,88,96,104,112,120,128,144,160,192,224])
        edges=torch.cat((torch.zeros(1),(r[:-1]+r[1:])/2,torch.tensor([240.])))
        area=math.pi*(edges[1:].square()-edges[:-1].square())/16
        angle=torch.arange(16)*2*math.pi/16
        self.register_buffer('r',r,persistent=False);self.register_buffer('area',area,persistent=False);self.register_buffer('angle',angle,persistent=False)
    def sample_margins(self,maps,qx,qy):
        shape=qx.shape;xx=qx.clamp(0,255);yy=qy.clamp(0,255)
        x0=xx.floor().long();y0=yy.floor().long();x1=(x0+1).clamp_max(255);y1=(y0+1).clamp_max(255)
        wx=xx-x0;wy=yy-y0;base=maps[:,None,None]*65536;field=self.margins.reshape(-1,2)
        def at(x,y):return field[(base+y*256+x).reshape(-1)].reshape(*shape,2).float()
        return at(x0,y0)*((1-wx)*(1-wy))[...,None]+at(x1,y0)*(wx*(1-wy))[...,None]+at(x0,y1)*((1-wx)*wy)[...,None]+at(x1,y1)*(wx*wy)[...,None]
    def nodes(self,x,maps):
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            x=x.float();info=self.map_info[maps];hm=info[:,0].long();tr=info[:,1].float();tc=info[:,2].float()
            dx=x[:,5]*128;dy=x[:,6]*128;dh=torch.sqrt(dx.square()+dy.square()).clamp_min(.25);ux=dx/dh;uy=dy/dh
            rx=tc+dx;ry=tr+dy;rz=x[:,0]*100+1;ht=x[:,31];frac=torch.remainder(ht,1)
            ca=self.angle.cos()[None,:,None];sa=self.angle.sin()[None,:,None];r=self.r[None,None]
            ox=r*(ux[:,None,None]*ca-uy[:,None,None]*sa);oy=r*(uy[:,None,None]*ca+ux[:,None,None]*sa)
            qx=rx[:,None,None]+ox;qy=ry[:,None,None]+oy
            inside=(qx>=0)&(qx<=255)&(qy>=0)&(qy<=255)
            h,sx,sy=self.sample(hm,qx,qy).unbind(-1);em=self.sample_margins(maps,qx,qy)
            norm=torch.sqrt(1+sx.square()+sy.square());rxz=rz[:,None,None]-(h+.05)
            dr=torch.sqrt(r.square()+rxz.square()).clamp_min(.25)
            coso=((sx*ox+sy*oy+rxz)/norm/dr).clamp(0,1)
            slope=(h-rz[:,None,None])/r
            prior=F.pad(slope.cummax(2).values[:,:,:-1],(1,0),value=-1e6)
            clearance=(slope+.05/r-prior)*r
            rxvis=torch.sigmoid(clearance/.05)*inside
            txvis3=torch.sigmoid(-em[...,0]/.03);txvis2=torch.sigmoid(-em[...,1]/.03)
            txvis=torch.sigmoid(-((1-frac[:,None,None])*em[...,0]+frac[:,None,None]*em[...,1])/.03)
            txz=ht[:,None,None]+3-frac[:,None,None]-(h+.05);txx=tc[:,None,None]-qx;txy=tr[:,None,None]-qy
            dt=torch.sqrt(txx.square()+txy.square()+txz.square()).clamp_min(.25)
            cosi=((-sx*txx-sy*txy+txz)/norm/dt).clamp(0,1)
            dz=rz-ht-3+frac;direct2=dx.square()+dy.square()+dz.square()
            geom=self.area[None,None]*norm*cosi*coso*direct2[:,None,None]/(math.pi*dt.square()*dr.square())
            geom=geom.clamp(0,1e8)*inside
            align=((-txx*(-ox)-txy*(-oy)-txz*rxz)/(dt*dr)+2*cosi*coso).clamp(-1,1)
            excess=(dt+dr-torch.sqrt(direct2)[:,None,None]).clamp_min(0)
            expand=lambda v:v.expand_as(h)
            features=[(h-ht[:,None,None])/50,(h-rz[:,None,None])/50,
              (sx*ux[:,None,None]+sy*uy[:,None,None])/5,(-sx*uy[:,None,None]+sy*ux[:,None,None])/5,
              expand(ca),expand(sa),torch.log1p(expand(r))/6,torch.log1p(dt)/6,torch.log1p(dr)/6,
              cosi,coso,align,align.clamp_min(0).pow(16),torch.log1p(excess)/6,
              txvis3,txvis2,txvis,rxvis,torch.asinh(em[...,0]/2)/3,torch.asinh(em[...,1]/2)/3,
              torch.asinh(clearance/2)/3,torch.log10(geom+1e-16)/8,
              ox/128,oy/128,inside.float(),torch.log1p(expand(self.area[None,None]))/8]
            node=torch.stack(features,-1).reshape(len(x),-1,26)
            weights=(geom*txvis*rxvis).reshape(len(x),-1)
            return node,weights,inside.reshape(len(x),-1)
    def forward(self,x,maps):
        z=self.global_blocks(F.silu(self.global_in(self.embedding(x))))
        features,prior,inside=self.nodes(x,maps)
        nodes=self.node_blocks(F.silu(self.node_in(features)))
        q=self.query(z).reshape(len(x),2,64);keys=self.key(nodes);values=self.value(nodes)
        with torch.autocast('cuda',enabled=False):
            scores=torch.einsum('nbc,nlc->nbl',q.float(),keys.float())/8
            if self.kind=='transport':
                weights=prior[:,None]+self.visibility_floor.exp()[None,:,None]
                scores=scores+torch.log(weights.clamp_min(1e-20))
            scores=scores.masked_fill(~inside[:,None],-1e4)
            context=torch.einsum('nbl,nlc->nbc',scores.softmax(-1),values.float())
        gate=self.gate(z).reshape(len(x),2,4).float().softmax(-1);out=[]
        for b,head in enumerate(self.decoders):
            hidden=head[:-1](torch.cat((z,context[:,b]),-1))
            with torch.autocast('cuda',enabled=False):delta=head[-1](hidden.float())
            out.append((delta*gate[:,b]).sum(-1))
        return torch.stack(out,-1)
