"""C3 independent residual/global U-Net. Predicts two excess-loss fields."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.skip=nn.Conv2d(cin,cout,1,bias=False) if cin!=cout else nn.Identity()
        self.body=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout))
    def forward(self,x):return F.silu(self.body(x)+self.skip(x))

class Model(nn.Module):
    def __init__(self,width=32):
        super().__init__();cs=[width,width*2,width*4,width*8,width*12]
        self.stem=nn.Conv2d(27,width,3,padding=1)
        self.enc=nn.ModuleList([Block(width,width)]+[Block(cs[i-1],cs[i]) for i in range(1,5)])
        self.global_blocks=nn.ModuleList([nn.TransformerEncoderLayer(cs[-1],8,cs[-1]*2,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.dec=nn.ModuleList([Block(cs[i]+cs[i-1],cs[i-1]) for i in range(4,0,-1)])
        self.head=nn.Sequential(nn.Conv2d(width+27,width*2,1),nn.SiLU(),nn.Conv2d(width*2,2,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        original=x.float()
        with torch.autocast('cuda',enabled=False):x=self.stem(original)
        skips=[]
        for i,block in enumerate(self.enc):
            if i:x=F.avg_pool2d(x,2)
            x=block(x);skips.append(x)
        b,c,h,w=x.shape;t=x.flatten(2).transpose(1,2)
        for layer in self.global_blocks:t=layer(t)
        x=t.transpose(1,2).reshape(b,c,h,w)
        for block,skip in zip(self.dec,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=block(torch.cat((x,skip),1))
        with torch.autocast('cuda',enabled=False):return self.head(torch.cat((x.float(),original),1))*20
