"""Learn a latent field on both receiver triangles, then interpolate onto the released raster."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

class NativeMap(nn.Module):
    def __init__(self,channels,width,late=True):
        from src.c67_dense import ResidualMap
        super().__init__();self.late=late
        self.face_channels=(channels-37)//2
        assert self.face_channels==47
        rec=np.load(Path(__file__).with_name('c70_centroid_lookup0.npz'))
        self.register_buffer('indices',torch.as_tensor(rec['ids'].copy(),dtype=torch.long).T.contiguous())
        self.register_buffer('weights',torch.as_tensor(rec['weights'].copy(),dtype=torch.float32).T.contiguous())
        self.face=nn.Sequential(nn.Conv2d(self.face_channels,width,1),nn.GELU(),nn.Conv2d(width,width,1),nn.GELU(),nn.Conv2d(width,18,1))
        self.spatial=ResidualMap(37+20,width,True)
    def mix(self,faces):
        b,t,c,h,w=faces.shape
        flat=faces.permute(0,2,1,3,4).reshape(b,c,-1)
        out=sum(flat.index_select(2,self.indices[j])*self.weights[j].to(flat.dtype)[None,None] for j in range(3))
        return out.reshape(b,c,256,256)
    def read_faces(self,features):
        latent=self.face(features)
        # Two learned corrections to relative band pathloss. The native baseline is a feature computed from H and TX.
        excess=((-2*features[:,33:35])+latent[:,:2].float()).clamp(-2.5,7.)
        return torch.cat((latent[:,2:].float(),excess,torch.exp(-4.605170186*excess)),1)
    def forward(self,x):
        b=x.shape[0]
        faces=x[:,37:,:255,:255].reshape(b,2,self.face_channels,255,255)
        if self.late:
            y=self.read_faces(faces.reshape(2*b,self.face_channels,255,255)).reshape(b,2,20,255,255)
            y=self.mix(y)
        else:
            y=self.read_faces(self.mix(faces))
        y=torch.cat((y[:,:18],-torch.log(y[:,18:].clamp_min(1e-15))/4.605170186),1)
        return self.spatial(torch.cat((x[:,:37],y),1).contiguous(memory_format=torch.channels_last))
