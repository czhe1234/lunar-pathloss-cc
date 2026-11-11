"""Independent full-information residual models; ordered frequency-aware profiles."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.c34_model import NumericEmbedding

def physical_profile(x):
    """All quantities derive only from existing H/TX descriptors, never target/mask."""
    with torch.autocast('cuda',enabled=False):
        x=x.float();t=(torch.arange(128,device=x.device,dtype=torch.float32)+.5)/128
        e=2*torch.sinh(3*x[:,104:232])
        frac=torch.remainder(x[:,31],1)
        alt=e+frac[:,None]*(1-t)
        horizontal=((x[:,5]*128).square()+(x[:,6]*128).square()).clamp_min(.0625)
        dz=x[:,7]*50;d=torch.sqrt(horizontal+dz.square()).clamp_min(.25)
        lam=torch.tensor([299792458/415e6,299792458/5.775e9],device=x.device)
        scale=torch.sqrt(2/(lam[None,:,None]*d[:,None,None]*t[None,None,:]*(1-t[None,None,:])))
        vn=e[:,None]*scale;va=alt[:,None]*scale
        def excess(z,delta):
            a=torch.sqrt(horizontal[:,None]*t.square()+(delta[:,None]*t+z).square())
            b=torch.sqrt(horizontal[:,None]*(1-t).square()+(delta[:,None]*(1-t)-z).square())
            direct=torch.sqrt(horizontal+delta.square())
            return (a+b-direct[:,None]).clamp_min(0)
        ex=excess(e,dz);exa=excess(alt,dz+frac)
        ph=2*math.pi*ex[:,None]/lam[None,:,None]
        pha=2*math.pi*exa[:,None]/lam[None,:,None]
        gradient=F.pad(e[:,1:]-e[:,:-1],(1,0))
        ch=[x[:,104:232],torch.asinh(alt/2)/3,torch.asinh(gradient/2)/3,
            t[None].expand_as(e)*2-1,torch.log1p(d)[:,None].expand_as(e)/6,
            torch.asinh(vn[:,0])/3,torch.asinh(vn[:,1])/3,
            torch.asinh(va[:,0])/3,torch.asinh(va[:,1])/3,torch.log1p(ex)/4,
            ph[:,0].sin(),ph[:,0].cos(),ph[:,1].sin(),ph[:,1].cos(),
            pha[:,0].sin(),pha[:,0].cos(),pha[:,1].sin(),pha[:,1].cos()]
        return torch.stack(ch,-1)

class DenseBlock(nn.Module):
    def __init__(self,w):
        super().__init__();self.norm=nn.LayerNorm(w);self.up=nn.Linear(w,w*2);self.down=nn.Linear(w,w)
    def forward(self,x):
        a,b=self.up(self.norm(x)).chunk(2,-1)
        return x+.2*self.down(F.silu(a)*b)

class AttentionBlock(nn.Module):
    def __init__(self,w=64):
        super().__init__();self.norm=nn.LayerNorm(w);self.qkv=nn.Linear(w,w*3);self.out=nn.Linear(w,w);self.dense=DenseBlock(w)
    def forward(self,x):
        B,L,W=x.shape
        q,k,v=self.qkv(self.norm(x)).reshape(B,L,3,4,W//4).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,L,W)
        return self.dense(x+.2*self.out(y))

class ProfileConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(18,48,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.GroupNorm(8,48),nn.Conv2d(48,48,(1,5),padding=(0,2*d),dilation=(1,d),groups=48),nn.Conv2d(48,96,1),nn.GLU(dim=1),nn.Conv2d(48,48,1)) for d in [1,2,4,8,16,32]])
        self.out=nn.Linear(48,64)
    def forward(self,seq):
        z=self.stem(seq.transpose(1,2).unsqueeze(2).contiguous(memory_format=torch.channels_last))
        for block in self.blocks:z=z+.2*block(z)
        return self.out(z.squeeze(2).transpose(1,2))

class Model(nn.Module):
    def __init__(self,kind,teacher_state):
        super().__init__();assert kind in ['flat','conv','attention'];self.kind=kind
        s=teacher_state
        self.embedding=NumericEmbedding('periodic',s['embedding.mean'],s['embedding.std'],s['embedding.edges'])
        self.embedding.load_state_dict({k[len('embedding.'):]:v for k,v in s.items() if k.startswith('embedding.')})
        self.global_in=nn.Linear(294*9,384);self.global_blocks=nn.Sequential(*[DenseBlock(384) for _ in range(3)])
        if kind=='flat':
            self.profile=nn.Sequential(nn.Flatten(1),nn.Linear(128*18,256),nn.SiLU(),nn.Linear(256,128))
        elif kind=='conv':self.profile=ProfileConv()
        else:
            self.patch=nn.Linear(4*18,64);self.position=nn.Parameter(torch.randn(32,64)*.02)
            self.profile=nn.Sequential(*[AttentionBlock() for _ in range(3)])
        self.query=nn.Linear(384,128);self.key=nn.Linear(64,64);self.value=nn.Linear(64,64)
        self.decoders=nn.ModuleList([nn.Sequential(nn.Linear(384+64,256),nn.SiLU(),nn.Linear(256,4)) for _ in range(2)])
        self.gate=nn.Linear(384,8)
        for h in self.decoders:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
    def forward(self,x):
        z=self.global_blocks(F.silu(self.global_in(self.embedding(x))))
        seq=physical_profile(x)
        if self.kind=='flat':context=self.profile(seq).reshape(len(x),2,64)
        else:
            if self.kind=='attention':tokens=self.profile(self.patch(seq.reshape(len(x),32,4*18))+self.position)
            else:tokens=self.profile(seq)
            q=self.query(z).reshape(len(x),2,64)
            keys=self.key(tokens);values=self.value(tokens)
            with torch.autocast('cuda',enabled=False):
                w=torch.einsum('nbc,nlc->nbl',q.float(),keys.float())/8
                context=torch.einsum('nbl,nlc->nbc',w.softmax(-1),values.float())
        gates=self.gate(z).reshape(len(x),2,4).float().softmax(-1)
        pred=[]
        for b,h in enumerate(self.decoders):
            hidden=h[:-1](torch.cat((z,context[:,b]),-1))
            with torch.autocast('cuda',enabled=False):delta=h[-1](hidden.float())
            pred.append((delta*gates[:,b]).sum(-1))
        return torch.stack(pred,-1)
