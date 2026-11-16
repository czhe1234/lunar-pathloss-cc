"""Full-terrain learned fields plus frozen C38 representation. Independent CC code."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.c34_model import Model as Teacher,NumericEmbedding
from src.c47_model import DenseBlock
from src.c46_model import TerrainEncoder,at_pixels
from src.c52_model import Model as Latent
class Model(nn.Module):
 global_features=Latent.global_features
 def __init__(self,state,variant,warm_global=None):
  super().__init__();assert variant in ['local','path'];self.variant=variant
  self.embedding=NumericEmbedding('periodic',state['embedding.mean'],state['embedding.std'],state['embedding.edges'])
  self.embedding.load_state_dict({k[len('embedding.'):]:v for k,v in state.items() if k.startswith('embedding.')});self.embedding.requires_grad_(False)
  self.backbone=Teacher('periodic',8,1024,state['embedding.mean'],state['embedding.std'],state['embedding.edges'],state['bias'][0]*20);self.backbone.load_state_dict(state);self.backbone.requires_grad_(False)
  self.global_in=nn.Linear(2048+294+4,384);self.global_blocks=nn.Sequential(*[DenseBlock(384) for _ in range(3)])
  if warm_global is not None:
   self.global_in.load_state_dict({k[len('global_in.'):]:v for k,v in warm_global.items() if k.startswith('global_in.')})
   self.global_blocks.load_state_dict({k[len('global_blocks.'):]:v for k,v in warm_global.items() if k.startswith('global_blocks.')})
  self.encoder=TerrainEncoder();self.local_to=nn.Linear(96,64)
  self.kv=nn.ModuleList([nn.Conv2d(c,128,1) for c in self.encoder.cs])
  self.query=nn.Linear(384,128);self.offset=nn.Linear(384,32)
  nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
  self.geom=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,64))
  self.scale=nn.Parameter(torch.randn(4,64)*.02);self.attout=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.LayerNorm(64))
  self.heads=nn.ModuleList([nn.Sequential(nn.Linear(512,256),nn.SiLU(),nn.Linear(256,4)) for _ in range(2)])
  self.gate=nn.Linear(384,8)
  for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
  a=torch.arange(8)*2*math.pi/8;r=torch.tensor([8.,16.,32.,64.,96.,128.,64.,32.])
  self.register_buffer('ring',torch.stack((a.cos()*r,a.sin()*r),-1),persistent=False)
  self.register_buffer('t',torch.linspace(.05,.95,8),persistent=False)
 def encode(self,image):
  fields,local=self.encoder(image)
  return {'local':local,'kv':[p(f) for p,f in zip(self.kv,fields)],'raw':image}
 def decode(self,state,x,pixels):
  B,Q,_=x.shape;z=self.global_features(x.reshape(-1,294)).reshape(B,Q,384)
  local=self.local_to(at_pixels(state['local'],pixels))
  if self.variant=='local':contexts=[torch.zeros_like(local),torch.zeros_like(local)]
  else:
   with torch.autocast('cuda',enabled=False):
    xf=x.float();tx=torch.stack((xf[:,:,33]*256,xf[:,:,32]*256),-1);rx=tx+xf[:,:,5:7]*128
    line=tx[:,:,None]+(rx-tx)[:,:,None]*self.t[None,None,:,None]
    refs=torch.cat((line,rx[:,:,None]+self.ring[None,None]),2)
   refs=(refs+32*torch.tanh(self.offset(z).float()).reshape(B,Q,16,2)).clamp(0,255)
   with torch.autocast('cuda',enabled=False):
    grid=refs/127.5-1
    geo=F.grid_sample(state['raw'][:,[0,3,4]].float(),grid,align_corners=True,padding_mode='border').permute(0,2,3,1)
    h=geo[...,0]*100;sx=geo[...,1]*5;sy=geo[...,2]*5
    rz=xf[:,:,0]*100+1;tz=xf[:,:,31]+3
    vr=refs-rx[:,:,None];vt=refs-tx[:,:,None];zr=h-rz[:,:,None];zt=h-tz[:,:,None]
    dr=torch.sqrt(vr.square().sum(-1)+zr.square()).clamp_min(.25);dt=torch.sqrt(vt.square().sum(-1)+zt.square()).clamp_min(.25)
    dist=torch.sqrt((rx-tx).square().sum(-1)+(rz-tz).square()).clamp_min(.25)
    t=(vt*(rx-tx)[:,:,None]).sum(-1)/(rx-tx).square().sum(-1).clamp_min(1)[:,:,None]
    clearance=h-(tz[:,:,None]+t*(rz-tz)[:,:,None]);norm=torch.sqrt(1+sx.square()+sy.square())
    ci=(sx*vt[...,0]+sy*vt[...,1]-zt)/norm/dt;co=(sx*vr[...,0]+sy*vr[...,1]-zr)/norm/dr
    g=torch.stack((vr[...,0]/128,vr[...,1]/128,zr/20,torch.log1p(dt)/6,torch.log1p(dr)/6,torch.log1p((dt+dr-dist[:,:,None]).clamp_min(0))/6,t.clamp(-2,3),torch.asinh(clearance/2)/3,ci,co),-1)
    sampled=[F.grid_sample(v.float(),grid,align_corners=True,padding_mode='border').permute(0,2,3,1) for v in state['kv']]
   gp=self.geom(g);keys=torch.cat([v[...,:64]+gp+self.scale[j] for j,v in enumerate(sampled)],2);values=torch.cat([v[...,64:] for v in sampled],2)
   query=self.query(z).reshape(B,Q,2,64)
   with torch.autocast('cuda',enabled=False):
    weight=torch.einsum('bqnc,bqkc->bqnk',query.float(),keys.float())/8
    context=torch.einsum('bqnk,bqkc->bqnc',weight.softmax(-1),values.float())
   contexts=[self.attout(context[:,:,j]) for j in range(2)]
  gate=self.gate(z).reshape(B,Q,2,4).float().softmax(-1);out=[]
  for b,head in enumerate(self.heads):
   hidden=head[:-1](torch.cat((z,local,contexts[b]),-1))
   with torch.autocast('cuda',enabled=False):d=head[-1](hidden.float())
   out.append((d*gate[:,:,b]).sum(-1))
  return torch.stack(out,-1)
 def forward(self,image,x,pixels):return self.decode(self.encode(image),x,pixels)
