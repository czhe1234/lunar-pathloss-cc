"""C151b active matched-coordinate branch: Cartesian versus TX-centered polar.

The common fresh trunk is bound only after verified C150/C151a results.
No historical trained parameters are loaded. Both coordinate modes use the same
branch parameters and same 256x192 sequence grid; only the coordinate mapping
changes. Global96+Local64+branch16 enter the common176->128 fusion.
"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from src.c150_field import Field as BlindField
from src.c151_field import Field as AlignedField
from src.c151_polar import TxRadialBranch, tx_from_inputs

BRANCH_CONFIG = {"width":32,"out_channels":16,"ntheta":256,"nr":192,"depth":2,"heads":4}
TRUNKS = ("blind","aligned")
PARAMETERS_BY_TRUNK = {"blind":15245906,"aligned":15237586}
MODES = ("cartesian","polar")


class Field(nn.Module):
    def __init__(self, trunk_kind, branch_mode, branch_config=None):
        super().__init__()
        if trunk_kind not in TRUNKS or branch_mode not in MODES:
            raise ValueError("Unknown common trunk or coordinate arm")
        self.trunk_kind = trunk_kind
        self.branch_mode = branch_mode
        self.branch_config = dict(BRANCH_CONFIG if branch_config is None else branch_config)
        if self.branch_config != BRANCH_CONFIG:
            raise ValueError("C151b branch budget is fixed")
        self.trunk = BlindField() if trunk_kind=="blind" else AlignedField("aligned_global")
        self.trunk.mix = nn.Conv2d(176,128,1)
        self.branch = TxRadialBranch(mode=branch_mode,**self.branch_config)

    def forward(self, x, map_ids):
        if x.ndim!=4 or x.shape[1:]!=(68,256,256) or len(map_ids)!=len(x):
            raise ValueError("C151b needs raw68/full256 fields and explicit ordered map IDs")
        base = self.trunk
        row = (-128*x[:,6,0,0]).round().long().clamp(0,255)
        col = (-128*x[:,5,0,0]).round().long().clamp(0,255)
        tx = x[torch.arange(len(x),device=x.device),:,row,col].clamp(-12,12)
        packed = base.pack(x) if self.trunk_kind=="blind" else base.pack(x,map_ids)
        z = base.stem(packed)
        a,bias = base.film(tx).to(z.dtype)[:,:,None,None].chunk(2,1)
        z = z*(1+.1*a.tanh())+.1*bias
        for block in base.blocks:
            z = block(z)
        global_field = F.pixel_shuffle(base.unpack(z),4)
        local_field = base.local(x.clamp(-12,12))
        radial = self.branch(x,tx_from_inputs(x)).to(global_field.dtype)
        if radial.shape[1:]!=(16,256,256):
            raise ValueError("The active coordinate branch must return sixteen full-size channels")
        fused = base.refine(base.mix(torch.cat((global_field,local_field,radial),1)))
        return 4*base.head(fused)


def create(trunk_kind, branch_mode, seed=123041, branch_config=None):
    torch.manual_seed(seed)
    return Field(trunk_kind,branch_mode,branch_config)


def digest(net):
    result=hashlib.sha256()
    for name,tensor in net.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()
