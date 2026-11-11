"""Native power-state correction transported through explicit fixed ray couplings."""
import torch
from torch import nn
from torch.nn import functional as F
from src.c133_model import TerrainContext,Law
class NativeState(nn.Module):
 def __init__(self,mode='native'):
  super().__init__();self.mode=mode;self.encoder=TerrainContext();self.state=Law(6);self.measure=Law(2)
  yy,xx=torch.meshgrid(torch.arange(255),torch.arange(255),indexing='ij');a=(yy*256+xx).reshape(-1)
  faces=torch.cat((torch.stack((a,a+1,a+257),1),torch.stack((a,a+257,a+256),1)),0)
  self.register_buffer('faces',faces)
 def make_context(self,h):
  with torch.autocast('cuda',torch.bfloat16):encoded=self.encoder(h)
  return encoded
 def map_forward(self,encoded,g,base):
  # encoded comes only from this terrain; TX affects source illumination/state.
  face=encoded.reshape(16,-1).T[self.faces].float().mean(1)
  tx=F.grid_sample(encoded[None].float(),g['txxy'].reshape(1,1,1,2),align_corners=True).reshape(16)
  inp=torch.cat((g['feat'].float(),face,tx.expand(len(face),16)),1)
  logs=self.state(inp)
  if self.mode=='coarse':
   a=logs.reshape(2,255,255,6).permute(0,3,1,2)
   a=F.avg_pool2d(F.pad(a,(0,1,0,1),mode='replicate'),8,stride=8)
   logs=F.interpolate(a,size=(255,255),mode='bilinear',align_corners=False).permute(0,2,3,1).reshape(-1,6)
  source=logs[:,:2].exp();qstate=logs[:,2:].reshape(-1,2).exp()
  ix=g['uf'].long()
  df=g.get('df',g['feat'][ix]).float()
  di=torch.cat((df,face[ix],tx.expand(len(ix),16)),1)
  dl=self.measure(di)
  gain=g['dw']*dl.exp()
  si=g['si'].long();ai=g['ai'].long()
  gain=gain.index_add(0,si[:,0],g['sw']*source[si[:,1]])
  gain=gain.index_add(0,ai[:,0],g['aw']*qstate[ai[:,1]])
  ratio=(gain[g['inv'].long()]*g['pw']).sum(1)
  ratio=torch.where(g['active'],ratio,torch.ones_like(ratio))
  return base-10/2.302585092994046*ratio.clamp_min(1e-12).log()
 def forward(self,heights,graphs,bases,inverse=None):
  enc=self.make_context(heights)
  if inverse is None:inverse=list(range(len(graphs)))
  return [self.map_forward(enc[inverse[j]],g,bases[j]) for j,g in enumerate(graphs)]
