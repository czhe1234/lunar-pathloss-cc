"""C24: widen the local readout while preserving its initial function."""
import torch
from torch import nn
from c21_model import PointRes
from c21_model_r1 import Model
def expand_readout(net,noise=.001):
    device=next(net.parameters()).device
    for b in range(2):
        old=net.readout[b]
        wide=nn.Sequential(nn.Linear(old[0].in_features,1024),nn.SiLU(),
                           PointRes(1024),PointRes(1024),nn.Linear(1024,1)).to(device)
        with torch.no_grad():
            wide[0].weight.copy_(old[0].weight.repeat(2,1))
            wide[0].bias.copy_(old[0].bias.repeat(2))
            for layer in [2,3]:
                for name in ["a","b"]:
                    source=getattr(old[layer],name);dest=getattr(wide[layer],name)
                    weights=source.weight.repeat(2,2)/2
                    # Duplicate inputs are equal at initialization, so opposite
                    # outgoing perturbations cancel while breaking gradient symmetry.
                    eps=torch.randn((1024,512),device=device)*noise
                    weights[:,:512]+=eps;weights[:,512:]-=eps
                    dest.weight.copy_(weights);dest.bias.copy_(source.bias.repeat(2))
            weights=old[-1].weight.repeat(1,2)/2
            eps=torch.randn((1,512),device=device)*noise
            weights[:,:512]+=eps;weights[:,512:]-=eps
            wide[-1].weight.copy_(weights);wide[-1].bias.copy_(old[-1].bias)
        net.readout[b]=wide
    return net
