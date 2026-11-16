"""C151a: equal-budget active spatial-alignment control for global physics.

The raw 68-channel teacher-free input is unchanged from C150. The global path
uses 13 derived terrain/TX fields plus 55 solver/diffraction fields. Both arms
have the same dense stem and active parameters; only global physics alignment
differs. Map IDs seed a fixed permutation, never a learned embedding.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.c123_lossless import Field as OriginalField, global_inputs

CHANNELS = 68
GLOBAL_CHANNELS = 68
PERMUTATION_SEED = 151017
PARAMETERS = 15214786
MODES = ("shuffled_global", "aligned_global")


class Field(OriginalField):
    def __init__(self, global_mode, permutation_seed=PERMUTATION_SEED):
        super().__init__("native")
        if global_mode not in MODES:
            raise ValueError("Unknown global-physics arm")
        self.global_mode = global_mode
        self.permutation_seed = int(permutation_seed)
        self.stem = nn.Conv2d(GLOBAL_CHANNELS*16, 384, 1)
        self.film = nn.Linear(CHANNELS, 768)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.local[0] = nn.Conv2d(CHANNELS, 64, 1)
        for block in self.blocks:
            block.ff.up = nn.Linear(384, 715)
            block.ff.down = nn.Linear(715, 384)
        for module, hidden in zip((self.local[1], self.local[2]), (119, 118)):
            module.up = nn.Linear(64, hidden)
            module.down = nn.Linear(hidden, 64)
        for module, hidden in zip((self.refine[0], self.refine[1]), (222, 221)):
            module.up = nn.Linear(128, hidden)
            module.down = nn.Linear(hidden, 128)
        self._permutation_cpu = {}

    def permutation(self, map_id):
        map_id = int(map_id)
        if map_id not in self._permutation_cpu:
            token = f"C151-global-spatial-permutation-v1|{self.permutation_seed}|{map_id}".encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
            order = np.random.default_rng(seed).permutation(65536).astype(np.int64, copy=False)
            self._permutation_cpu[map_id] = torch.from_numpy(order)
        return self._permutation_cpu[map_id]

    def global_fields(self, x, map_ids):
        if x.ndim != 4 or x.shape[1:] != (68,256,256) or len(map_ids) != len(x):
            raise ValueError("C151 needs raw68/full256 fields and explicit ordered map IDs")
        terrain = global_inputs(x)
        physics = torch.cat((x[:,11:56].float(),x[:,58:68].float()),1).clamp(-12,12)
        if self.global_mode == "shuffled_global":
            if torch.is_tensor(map_ids):
                raise TypeError("Pass the explicit CPU map-ID sequence; avoid implicit GPU synchronization")
            order = torch.stack([self.permutation(i) for i in map_ids]).to(x.device)
            physics = physics.flatten(2).gather(2, order[:,None].expand(-1,55,-1)).reshape_as(physics)
        return torch.cat((terrain,physics),1)

    def pack(self, x, map_ids):
        return F.pixel_unshuffle(self.global_fields(x,map_ids),4)

    def forward(self, x, map_ids):
        count = len(x)
        row = (-128*x[:,6,0,0]).round().long().clamp(0,255)
        col = (-128*x[:,5,0,0]).round().long().clamp(0,255)
        tx = x[torch.arange(count,device=x.device),:,row,col].clamp(-12,12)
        z = self.stem(self.pack(x,map_ids))
        a,bias = self.film(tx).to(z.dtype)[:,:,None,None].chunk(2,1)
        z = z*(1+.1*a.tanh())+.1*bias
        for block in self.blocks:
            z = block(z)
        z = F.pixel_shuffle(self.unpack(z),4)
        local = self.local(x.clamp(-12,12))
        z = self.refine(self.mix(torch.cat((z,local),1)))
        return 4*self.head(z)


def create(global_mode, seed=123041, permutation_seed=PERMUTATION_SEED):
    torch.manual_seed(seed)
    return Field(global_mode,permutation_seed)


def digest(net):
    value = hashlib.sha256()
    for name,tensor in net.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()
