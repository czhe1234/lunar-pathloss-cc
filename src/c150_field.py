"""C150 teacher-free Field: shared diffraction inputs; global terrain path stays blind.

Raw channels are 11 H/TX, 45 frozen solver, two literal-zero former C82 slots,
then 10 normalized deterministic diffraction channels. No labels/masks enter.
"""
import hashlib
import torch
from torch import nn
from src.c123_lossless import Field as OriginalField

CHANNELS = 68
DIFFRACTION_CHANNELS = 10


class Field(OriginalField):
    def __init__(self):
        super().__init__("native")
        self.film = nn.Linear(CHANNELS, 768)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.local[0] = nn.Conv2d(CHANNELS, 64, 1)


def create(seed=123041):
    torch.manual_seed(seed)
    return Field()


def digest(net):
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def normalize_diffraction(raw):
    """Input-only fixed analytic transforms; raw is [B,10,H,W], Float32."""
    if raw.ndim != 4 or raw.shape[1] != 10:
        raise ValueError("Expected ten raw diffraction channels")
    values = []
    for offset in (0, 5):
        nu, position, integral, deygout, bullington = raw[:, offset:offset+5].unbind(1)
        values.extend((torch.asinh(nu)/3, 2*position-1,
                       torch.log1p(integral.clamp_min(0))/3, deygout/20, bullington/20))
    return torch.stack(values, 1)


def inputs(basic, physical, normalized_diffraction):
    if basic.shape[1] != 11 or physical.shape[1] != 45:
        raise ValueError("Frozen basic/solver channel contract changed")
    if normalized_diffraction.shape[1] != 10:
        raise ValueError("Expected ten normalized diffraction channels")
    zero = torch.zeros_like(basic[:, :2])
    return torch.cat((basic, physical, zero, normalized_diffraction), 1).contiguous(
        memory_format=torch.channels_last)
