"""Independent numeric embeddings and shared-weight ensemble for C10 inputs.
Architecture ideas: arXiv:2203.05556 and arXiv:2410.24210. No external code.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class NumericEmbedding(nn.Module):
 def __init__(self,mode,mean,std,edges,outdim=8):
  super().__init__()
  self.mode=mode
  self.register_buffer('mean',mean.float())
  self.register_buffer('std',std.float())
  self.register_buffer('edges',edges.float())
  d=len(mean);self.weight=nn.Parameter(torch.randn(d,16,outdim)/4)
  self.bias=nn.Parameter(torch.zeros(d,outdim))
  if mode=='periodic': self.frequencies=nn.Parameter(torch.randn(d,8)*.1)
 def forward(self,x):
  with torch.autocast('cuda',enabled=False):
   z=((x.float()-self.mean)/self.std).clamp(-20,20)
   if self.mode=='ple':
    e=((z[:,:,None]-self.edges[None,:,:-1])/(self.edges[None,:,1:]-self.edges[None,:,:-1]).clamp_min(1e-5)).clamp(0,1)
   else:
    angle=2*math.pi*z[:,:,None]*self.frequencies[None]
    e=torch.cat((angle.sin(),angle.cos()),-1)
   h=F.relu(torch.einsum('ndq,dqe->nde',e,self.weight)+self.bias)
   return torch.cat((z,h.flatten(1)),1)

class EnsembleLinear(nn.Module):
 def __init__(self,din,dout,k,first=False):
  super().__init__();self.k=k
  self.weight=nn.Parameter(torch.empty(dout,din))
  nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
  self.r=nn.Parameter(torch.randn(k,din)*.5 if first and k>1 else torch.ones(k,din))
  self.s=nn.Parameter(torch.ones(k,dout))
  self.bias=nn.Parameter(torch.zeros(k,dout))
 def forward(self,x):
  return F.linear(x*self.r,self.weight)*self.s+self.bias

class Model(nn.Module):
 def __init__(self,mode,k,width,mean,std,edges,target_mean):
  super().__init__();self.k=k
  self.embedding=NumericEmbedding(mode,mean,std,edges)
  self.layers=nn.ModuleList([EnsembleLinear(len(mean)*9,width,k,True)]+[EnsembleLinear(width,width,k) for _ in range(2)])
  self.head=nn.Parameter(torch.empty(k,width,2));nn.init.normal_(self.head,std=.005)
  self.bias=nn.Parameter(target_mean[None].repeat(k,1)/20)
 def forward(self,x):
  z=self.embedding(x)[:,None].expand(-1,self.k,-1)
  for layer in self.layers:
   z=F.dropout(F.relu(layer(z)),.05,self.training)
  with torch.autocast('cuda',enabled=False):
   return torch.einsum('nkw,kwo->nko',z.float(),self.head)+self.bias
