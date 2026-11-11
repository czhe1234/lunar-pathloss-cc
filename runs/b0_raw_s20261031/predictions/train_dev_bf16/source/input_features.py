"""Inference-only input features. This module never opens pathloss or validity masks."""
import torch
from torch.nn import functional as F

PROFILE_SAMPLES = 192

def terrain_profiles(h, r, c, samples=PROFILE_SAMPLES):
    b = len(h)
    yy, xx = torch.meshgrid(torch.arange(256,device=h.device,dtype=torch.float32),torch.arange(256,device=h.device,dtype=torch.float32),indexing='ij')
    yy=yy.flatten();xx=xx.flatten()
    t=torch.linspace(0,1,samples+2,device=h.device)[1:-1]
    px=c[:,None,None]+(xx[None,:,None]-c[:,None,None])*t[None,None,:]
    py=r[:,None,None]+(yy[None,:,None]-r[:,None,None])*t[None,None,:]
    grid=torch.stack((px/127.5-1.,py/127.5-1.),dim=-1)
    terrain=F.grid_sample(h[:,None],grid,mode='bilinear',padding_mode='border',align_corners=True)[:,0]
    txh=h[torch.arange(b,device=h.device),r,c]+3.
    receiver=h.flatten(1)+1.
    straight=txh[:,None,None]+(receiver[:,:,None]-txh[:,None,None])*t[None,None,:]
    excess=terrain-straight
    features=torch.stack((excess.amax(-1),excess.clamp_min(0).mean(-1)),dim=1).reshape(b,2,256,256)/100.
    return features.half().float()  # Match the float16 training feature cache.

def make_inputs(h,r,c,mode='raw',profile=None):
    tx=torch.zeros_like(h)
    tx.flatten(1).scatter_(1,(r*256+c)[:,None],1.)
    features=[h/100.,tx]
    if mode in ('geometry','profile'):
        yy,xx=torch.meshgrid(torch.arange(256,device=h.device,dtype=torch.float32),torch.arange(256,device=h.device,dtype=torch.float32),indexing='ij')
        dx=xx[None]-c[:,None,None];dy=yy[None]-r[:,None,None]
        txh=h[torch.arange(len(h),device=h.device),r,c]
        dz=h+1.-(txh[:,None,None]+3.)
        distance=torch.sqrt(dx.square()+dy.square()+dz.square())
        features += [dx/256.,dy/256.,torch.log1p(distance)/6.,dz/100.]
    if mode=='profile':
        if profile is None:profile=terrain_profiles(h,r,c)
        features += [profile[:,0],profile[:,1]]
    return torch.stack(features,1).contiguous(memory_format=torch.channels_last)
