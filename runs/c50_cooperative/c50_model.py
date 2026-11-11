"""Independent nonlinear experts warm-started exactly from C38 branches."""
import torch
from torch import nn
from torch.nn import functional as F
from src.c34_model import NumericEmbedding
class ExpertLinear(nn.Module):
    def __init__(self,state,prefix,k):
        super().__init__()
        self.weight=nn.Parameter(state[prefix+'weight'][None].repeat(k,1,1).clone())
        self.r=nn.Parameter(state[prefix+'r'].clone());self.s=nn.Parameter(state[prefix+'s'].clone());self.bias=nn.Parameter(state[prefix+'bias'].clone())
    def forward(self,x):
        v=torch.bmm((x*self.r).permute(1,0,2),self.weight.transpose(1,2)).permute(1,0,2)
        return v*self.s+self.bias
class Model(nn.Module):
    def __init__(self,state,dropout=.02):
        super().__init__();self.k=state['head'].shape[0];self.dropout=dropout
        self.embedding=NumericEmbedding('periodic',state['embedding.mean'],state['embedding.std'],state['embedding.edges'])
        self.embedding.load_state_dict({k[len('embedding.'):]:v for k,v in state.items() if k.startswith('embedding.')})
        self.layers=nn.ModuleList([ExpertLinear(state,f'layers.{i}.',self.k) for i in range(3)])
        self.head=nn.Parameter(state['head'].clone());self.bias=nn.Parameter(state['bias'].clone())
        self.gate=nn.Sequential(nn.Linear(294,64),nn.SiLU(),nn.Linear(64,self.k*2))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
    def forward(self,x):
        embedded=self.embedding(x);z=embedded[:,None].expand(-1,self.k,-1)
        for layer in self.layers:z=F.dropout(F.relu(layer(z)),self.dropout,self.training)
        g=self.gate(embedded[:,:294])
        with torch.autocast('cuda',enabled=False):
            q=torch.einsum('nkw,kwo->nko',z.float(),self.head)+self.bias
            gates=g.float().reshape(len(x),self.k,2).softmax(1)
            return (q*gates).sum(1)
