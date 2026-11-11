"""C23: input-conditioned readout modulation; constant-context matched control."""
import torch
from torch import nn
from torch.nn import functional as F
from c21_model_r1 import Model as Parent
class Model(Parent):
    def __init__(self,mean,std,conditional=True):
        super().__init__(mean,std,geometry=False)
        self.conditional=bool(conditional)
        self.condition=nn.Sequential(nn.LayerNorm(189),nn.Linear(189,256),nn.SiLU(),nn.Linear(256,2048))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
    def encode(self,x,tx):
        state=super().encode(x,tx)
        local,z,geo,kv,tx,raw=state
        context=torch.cat((local.float().mean((2,3)),z.mean((2,3)),
                           z.std((2,3),correction=0),
                           tx/tx.new_tensor([256.,256.,100.])),dim=1)
        if not self.conditional:context=torch.zeros_like(context)
        mods=self.condition(context).reshape(len(x),2,2,512)
        return state+(mods,)
    def decode(self,state,pixels,chunk=512):
        loc,z,geo,kv,tx,raw,mods=state;B=len(z);out=[]
        for lo in range(0,pixels.shape[1],chunk):
            pix=pixels[:,lo:lo+chunk];Q=pix.shape[1]
            local=loc.flatten(2).gather(2,pix[:,None].expand(-1,loc.shape[1],-1)).transpose(1,2)
            orig=z.flatten(2).gather(2,pix[:,None].expand(-1,45,-1)).transpose(1,2)
            rr=(pix//256).float();cc=(pix%256).float()
            coords=torch.stack((rr/256,cc/256,(rr-tx[:,None,0])/256,(cc-tx[:,None,1])/256),-1)
            angle=coords[:,:,:,None]*self.frequencies
            pos=torch.cat((angle.sin(),angle.cos()),-1).flatten(2)
            bands=[]
            for b in [0,1]:
                q=self.query(torch.cat((local,orig,pos,self.freq[b][None,None].expand(B,Q,-1)),-1))
                qq=self.qnorm(q).reshape(B,Q,self.heads,-1).transpose(1,2)
                attended=F.scaled_dot_product_attention(qq,kv[0],kv[1],dropout_p=0.)
                q=q+self.attout(attended.transpose(1,2).reshape(B,Q,-1));q=q+self.ff(q)
                hidden=self.readout[b][:-1](torch.cat((local,orig,pos,q),-1))
                hidden=hidden*(1+.5*mods[:,b,0,None].tanh())+.5*mods[:,b,1,None]
                with torch.autocast("cuda",enabled=False):
                    pred=self.readout[b][-1](hidden.float()).squeeze(-1)
                bands.append(pred)
            out.append(torch.stack(bands,-1))
        return torch.cat(out,1)
