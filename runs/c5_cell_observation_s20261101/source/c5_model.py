"""C5: learned observation of a subpixel visibility/power mixture.

The visibility calibration is a latent observation correction, NOT a claim that
actual TX height differs from the official 3 m. No mask or target enters forward.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from c4_model import Model as C4Model


def mix_losses_db(direct_db, blocked_db, visibility, power_fraction):
    """Area mixture in power, optionally followed by dB-domain interpolation."""
    c=math.log(10.)/10.
    v=visibility.clamp(0,1)
    power=-torch.logaddexp(torch.log(v.clamp_min(1e-30))-c*direct_db,
                          torch.log((1-v).clamp_min(1e-30))-c*blocked_db)/c
    db=v*direct_db+(1-v)*blocked_db
    return power_fraction*power+(1-power_fraction)*db


class Model(C4Model):
    def __init__(self,width=32):
        super().__init__(width)
        self.observation=nn.Sequential(nn.Conv2d(width+31+2,96,1),nn.SiLU(),
                                      nn.Conv2d(96,64,1),nn.SiLU(),nn.Conv2d(64,9,1))
        nn.init.zeros_(self.observation[-1].weight)
        nn.init.zeros_(self.observation[-1].bias)
        yy,xx=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
        self.register_buffer('pixel_xy',torch.stack((xx,yy),-1),persistent=False)
        # 3x3 midpoint quadrature of a one-metre footprint.
        q=torch.tensor([-1/3,0.,1/3]);oy,ox=torch.meshgrid(q,q,indexing='ij')
        self.register_buffer('quadrature_xy',torch.stack((ox,oy),-1).reshape(9,2),persistent=False)

    def observation_visibility(self,original,q):
        # Decode existing input-only geometry without re-estimating it from labels.
        e3=(2*torch.sinh(3*original[:,11:19])).amax(1)
        e2=2*torch.sinh(3*original[:,19])
        # Three learned corrections: visibility threshold, subpixel registration,
        # transition width. All are bounded and shared by both frequency bands.
        calibration=1.5*torch.tanh(q[:,0:1])
        shift=.75*torch.tanh(q[:,1:3]).permute(0,2,3,1)[:,None]
        xy=self.pixel_xy[None,None]+self.quadrature_xy[None,:,None,None,:]+shift
        grid=(xy/127.5-1).reshape(len(original),9*256,256,2)
        fields=torch.stack((e3,e2-e3),1)
        samples=F.grid_sample(fields,grid,mode='bilinear',padding_mode='border',align_corners=True).reshape(len(original),2,9,256,256)
        margin=samples[:,0]+calibration*samples[:,1]
        temperature=.01+.09*torch.sigmoid(q[:,3:4])
        return torch.sigmoid(-margin/temperature).mean(1,keepdim=True)

    def forward(self,x,return_diagnostics=False):
        original=x.float()
        with torch.autocast('cuda',enabled=False):z=self.stem(original)
        skips=[]
        for i,block in enumerate(self.enc):
            if i:z=F.avg_pool2d(z,2)
            z=block(z);skips.append(z)
        b,c,h,w=z.shape;t=z.flatten(2).transpose(1,2)
        for layer in self.global_blocks:t=layer(t)
        z=t.transpose(1,2).reshape(b,c,h,w)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            z=F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            z=block(torch.cat((z,skip),1))
        with torch.autocast('cuda',enabled=False):
            features=torch.cat((z.float(),original),1)
            base=self.head(features)*20
        # Hidden projections use BF16; final scalar observation operations use FP32.
        hidden=self.observation[:-1](torch.cat((features,base/20),1))
        with torch.autocast('cuda',enabled=False):
            q=self.observation[-1](hidden.float())
            v=self.observation_visibility(original,q)
            direct=3*q[:,5:7]
            blocked=base+8*q[:,7:9]
            fraction=torch.sigmoid(q[:,4:5])
            pred=mix_losses_db(direct,blocked,v,fraction)
        if return_diagnostics:
            return pred,{'visibility':v,'direct_excess':direct,'blocked_excess':blocked,
                         'power_fraction':fraction,'latent_visibility_calibration':1.5*torch.tanh(q[:,0:1])}
        return pred
