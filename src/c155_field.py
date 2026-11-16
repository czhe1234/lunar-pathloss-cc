"""C155 full-resolution four-level ConvNeXt U-Net; raw68 -> residual2."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

WIDTHS=(48,96,192,256)
DEPTH=2
PARAMETERS=3193650

class LayerNorm2d(nn.Module):
    def __init__(self,channels):
        super().__init__();self.weight=nn.Parameter(torch.ones(channels));self.bias=nn.Parameter(torch.zeros(channels))
    def forward(self,x):
        x=F.layer_norm(x.permute(0,2,3,1),(x.shape[1],),self.weight,self.bias,1e-6)
        return x.permute(0,3,1,2)

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.dw=nn.Conv2d(c,c,7,padding=3,groups=c)
        self.norm=LayerNorm2d(c)
        self.up=nn.Conv2d(c,4*c,1);self.down=nn.Conv2d(4*c,c,1)
        self.gamma=nn.Parameter(torch.full((1,c,1,1),1e-6))
    def forward(self,x):
        z=self.down(F.gelu(self.up(self.norm(self.dw(x)))))
        return x+self.gamma*z

class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(68,WIDTHS[0],3,padding=1)
        self.encoder=nn.ModuleList([nn.Sequential(*(Block(c) for _ in range(DEPTH))) for c in WIDTHS])
        self.down=nn.ModuleList([nn.Sequential(LayerNorm2d(a),nn.Conv2d(a,b,2,stride=2))
            for a,b in zip(WIDTHS[:-1],WIDTHS[1:])])
        self.up=nn.ModuleList([nn.Conv2d(WIDTHS[j+1],WIDTHS[j],1) for j in (2,1,0)])
        self.merge=nn.ModuleList([nn.Conv2d(2*WIDTHS[j],WIDTHS[j],1) for j in (2,1,0)])
        self.decoder=nn.ModuleList([nn.Sequential(*(Block(WIDTHS[j]) for _ in range(DEPTH))) for j in (2,1,0)])
        self.head=nn.Conv2d(WIDTHS[0],2,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,map_ids=None):
        if x.ndim!=4 or x.shape[1:]!=(68,256,256):raise ValueError("Exact raw68/256x256 contract required")
        x=x.clamp(-12,12);skips=[]
        z=self.stem(x)
        for j,block in enumerate(self.encoder):
            z=block(z);skips.append(z)
            if j<3:z=self.down[j](z)
        for j,(up,merge,block) in enumerate(zip(self.up,self.merge,self.decoder)):
            skip=skips[2-j]
            z=up(F.interpolate(z,size=skip.shape[-2:],mode="bilinear",align_corners=False))
            z=block(merge(torch.cat((z,skip),1)))
        return 4*self.head(z)

def create(seed=123041):
    torch.manual_seed(seed);return Field()

def digest(net):
    h=hashlib.sha256()
    for name,tensor in net.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
