"""C27: preserve ordered public-terrain profiles at the point decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from c21_model_r1 import Model as BaseModel

class Model(BaseModel):
 def __init__(self,mean,std,mode="fine"):
  super().__init__(mean,std,geometry=False)
  assert mode in ("none","coarse","fine");self.profile_mode=mode
  self.path_encoder=nn.Sequential(nn.Conv1d(6,16,5,stride=2,padding=2),nn.SiLU(),
   nn.Conv1d(16,32,5,stride=2,padding=2),nn.SiLU(),
   nn.Conv1d(32,64,5,stride=2,padding=2),nn.SiLU(),
   nn.Conv1d(64,64,5,stride=2,padding=2),nn.SiLU(),nn.Flatten(),
   nn.Linear(1024,128),nn.SiLU(),nn.Linear(128,1024))
  nn.init.zeros_(self.path_encoder[-1].weight);nn.init.zeros_(self.path_encoder[-1].bias)
  self.register_buffer("path_t",(torch.arange(256,dtype=torch.float32)+.5)/256,persistent=False)

 def profile(self,raw,tx,pix):
  # Public H geometry uses FP32, before learned spatial downsampling.
  with torch.autocast(device_type=raw.device.type,enabled=False):
   H=raw[:,0:1].float()*100;tx=tx.float();B,Q=pix.shape
   rr=(pix//256).float();cc=(pix%256).float()
   dr=rr-tx[:,None,0];dc=cc-tx[:,None,1]
   t=self.path_t[None,None]
   yr=tx[:,None,None,0]+dr[:,:,None]*t
   xc=tx[:,None,None,1]+dc[:,:,None]*t
   grid=torch.stack((xc/127.5-1,yr/127.5-1),-1)
   hp=F.grid_sample(H,grid,mode="bilinear",padding_mode="border",align_corners=True)[:,0]
   hr=H[:,0].flatten(1).gather(1,pix)+1
   ht=tx[:,2]-3
   line=tx[:,None,None,2]+t*(hr[:,:,None]-tx[:,None,None,2])
   proxy=torch.floor(ht)+3
   proxyline=proxy[:,None,None]+t*(hr[:,:,None]-proxy[:,None,None])
   distance=torch.sqrt(dr.square()+dc.square()).clamp_min(.25)
   dh=F.pad(hp[:,:,1:]-hp[:,:,:-1],(0,1),mode="replicate")
   slope=dh/(distance[:,:,None]/256).clamp_min(1e-3)
   terrain=torch.stack((torch.asinh((hp-line)/2)/3,
    torch.asinh((hp-proxyline)/2)/3,
    torch.asinh((hp-ht[:,None,None])/10)/3,torch.asinh(slope)/3),2).clamp(-8,8)
   if self.profile_mode=="none":terrain=torch.zeros_like(terrain)
   elif self.profile_mode=="coarse":
    v=terrain.reshape(B*Q,4,256)
    terrain=F.interpolate(F.avg_pool1d(v,32),size=256,mode="linear",align_corners=False).reshape(B,Q,4,256)
   tt=t.expand(B,Q,-1)[:,:,None]
   dd=(torch.log1p(distance)/6)[:,:,None,None].expand(B,Q,1,256)
   return torch.cat((terrain,tt,dd),2).reshape(B*Q,6,256)

 def path_hidden(self,raw,tx,pix):
  B,Q=pix.shape
  return self.path_encoder(self.profile(raw,tx,pix)).reshape(B,Q,2,512)

 def decode(self,state,pixels,chunk=256):
  loc,z,geo,kv,tx,raw=state;B=len(z);out=[]
  for lo in range(0,pixels.shape[1],chunk):
   pix=pixels[:,lo:lo+chunk];Q=pix.shape[1]
   path=self.path_hidden(raw,tx,pix)
   local=loc.flatten(2).gather(2,pix[:,None].expand(-1,loc.shape[1],-1)).transpose(1,2)
   orig=z.flatten(2).gather(2,pix[:,None].expand(-1,45,-1)).transpose(1,2)
   rr=(pix//256).float();cc=(pix%256).float()
   coords=torch.stack((rr/256,cc/256,(rr-tx[:,None,0])/256,(cc-tx[:,None,1])/256),-1)
   angle=coords[:,:,:,None]*self.frequencies;pos=torch.cat((angle.sin(),angle.cos()),-1).flatten(2)
   g=self.geometry_features(pix,geo,tx,raw) if self.geometry else None
   bands=[]
   for b in [0,1]:
    q=self.query(torch.cat((local,orig,pos,self.freq[b][None,None].expand(B,Q,-1)),-1));qq=self.qnorm(q).reshape(B,Q,self.heads,-1).transpose(1,2)
    bias=None
    if self.geometry:
     with torch.autocast('cuda',enabled=False):bias=torch.einsum('bqkf,hf->bhqk',g,self.geoweight[b].float())
     bias=bias.to(qq.dtype)
    attended=F.scaled_dot_product_attention(qq,kv[0],kv[1],attn_mask=bias,dropout_p=0.)
    q=q+self.attout(attended.transpose(1,2).reshape(B,Q,-1));q=q+self.ff(q)
    hidden=self.readout[b][:-1](torch.cat((local,orig,pos,q),-1))
    with torch.autocast('cuda',enabled=False):pred=self.readout[b][-1](hidden.float()+path[:,:,b].float()).squeeze(-1)
    bands.append(pred)
   out.append(torch.stack(bands,-1))
  return torch.cat(out,1)
