"""Maps of directly visible terrain drive two learned exchanges per input tile."""
import ctypes
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.c70_field import FieldMap
ROOT=Path(__file__).resolve().parents[1]
class Visibility:
    def __init__(self):
        self.lib=ctypes.CDLL(str(ROOT/'src/libc73_visibility.so'));p=ctypes.c_void_p
        self.call=self.lib.visibility_graph73;self.call.argtypes=[p,p,p,ctypes.c_int,p,p,p];self.call.restype=ctypes.c_int
    @torch.no_grad()
    def __call__(self,h,tr,tc):
        # Geometry and edge intersection use full-precision released heights.
        b=len(h);a=h[:,:-1,:-1].flatten(1);r=h[:,:-1,1:].flatten(1);u=h[:,1:,:-1].flatten(1);d=h[:,1:,1:].flatten(1)
        z=torch.cat((a,a),1);sx=torch.cat((r-a,d-u),1);sy=torch.cat((d-r,u-a),1)
        g=torch.stack((z,sx,sy,torch.ones_like(z)),2).contiguous();height=h[torch.arange(b,device=h.device),(tr+1).clamp_max(255),tc].floor()+3
        tx=torch.stack((tc*255./256.,(tr+1)*255./256.,height),1).float().contiguous();top=h.amax((1,2)).contiguous()
        ix=torch.empty((b,2,4096,40),device=h.device,dtype=torch.int32);feat=torch.empty((*ix.shape,4),device=h.device)
        code=self.call(g.data_ptr(),tx.data_ptr(),top.data_ptr(),b,ix.data_ptr(),feat.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(f'visibility edge kernel error {code}')
        return ix,feat
class Exchange(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.q=nn.Linear(c,12);self.kv=nn.Linear(c,12+16)
        self.arc=nn.Sequential(nn.Linear(4,12),nn.SiLU(),nn.Linear(12,4))
        self.out=nn.Linear(17,c);self.scale=nn.Parameter(torch.full((c,),.1))
    def forward(self,receiver,source,adj,edge,shuffle):
        b,c,h,w=source.shape
        q=self.q(receiver.flatten(2).transpose(1,2)).reshape(b,4096,4,3)
        tokens=self.kv(source.flatten(2).transpose(1,2))
        valid=adj>=0;index=adj.clamp_min(0).long()
        if shuffle:index=(index*2053+1213)%4096
        neighbors=tokens[torch.arange(b,device=source.device)[:,None,None],index]
        key=neighbors[...,:12].reshape(b,4096,40,4,3);value=neighbors[...,12:].reshape(b,4096,40,4,4)
        weight=(key*q[:,:,None]).sum(-1)*.57735+self.arc(edge)
        weight=weight.float().masked_fill(~valid[:,:,:,None],-10000.).softmax(2)*valid[:,:,:,None]
        msg=(value*weight[:,:,:,:,None].to(value.dtype)).sum(2).reshape(b,4096,16)
        support=valid.float().mean(2,keepdim=True).to(value.dtype)
        y=self.out(torch.cat((msg,support),2))*self.scale
        return receiver+y.transpose(1,2).reshape(b,c,h,w)
class GraphField(FieldMap):
    def __init__(self,channels=56,width=40,shuffle=False):
        super().__init__(channels,width);self.shuffle=shuffle
        self.graph=Visibility();self.ground=Exchange(width+24);self.receiver=Exchange(width+24)
    def forward(self,x,height,tr,tc):
        # The two networks share shapes, initial weights and geometry inputs. Only source-node identity changes.
        ids,edges=self.graph(height,tr,tc)
        v=self.stage[0](self.stem(x));skip=[v]
        for j,down in enumerate(self.down):
            v=self.stage[j+1](down(v))
            if j==1:
                surface=self.ground(v,v,ids[:,0],edges[:,0],self.shuffle)
                v=self.receiver(v,surface,ids[:,1],edges[:,1],self.shuffle)
            skip.append(v)
        v=self.corridor(v,tr.float()/127.5-1,tc.float()/127.5-1)
        for j in (3,2,1,0):
            v=self.dec[j](skip[j]+F.interpolate(self.up[j](v),size=skip[j].shape[-2:],mode='bilinear',align_corners=False))
        return 4*self.head(v)
