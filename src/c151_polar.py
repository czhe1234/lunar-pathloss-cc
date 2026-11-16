"""C151b paired Cartesian/TX-polar feature branch (no labels or validity inputs).

Both modes use exactly the same active weights, 256 sequences of length 192,
two bidirectional radial Conv/SDPA/FFN blocks, and a nonzero 16-channel head.
Only coordinate routing differs. The default grid is not a lossless transform:
near a map corner its radial spacing is about 1.9 m and its far-radius angular
arc spacing about 8.8 m. The existing full-resolution paths remain necessary.

Input x follows C150's 68-channel teacher-free contract. TX coordinates passed
to forward are (row, col) in *receiver query image indices*, never raw integer
TX pixels. registered_query_tx converts published TX row/col, including the
registered +1 row offset. Do not clamp its result: edge TXs can be outside the
query image by a fraction of a pixel.

Native h/100 and central-gradient sx/5, sy/5 are first projected to the same
anti0 piecewise-triangle query grid used by C149/C150. Slopes are descriptors,
not exact face normals. Existing C73 and diffraction fields already live on the
query image. Geometric support is generated from coordinates only. It never
uses label validity, residuals, labels, C82, or map IDs.

Polar radii span 0..Rmax with one Rmax per TX, its maximum distance to the four
query-image corners. Theta is periodic, including inverse interpolation across
the seam. Cartesian control covers the image rectangle with identical tensor
and sequence sizes; sequences are bidirectional in both modes. There is no
angular convolution or learned coordinate embedding that differs by mode.
"""
import argparse
from dataclasses import dataclass
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 68
PAYLOAD_CHANNELS = 15
OUT_CHANNELS = 16
MODES = ("cartesian", "polar")
INVERSE_SUPPORT_THRESHOLD = 0.25
QUERY_ORIGIN = 1.0/3.0
QUERY_STEP = (255.0-2.0/3.0)/255.0
METERS_PER_QUERY_PIXEL = QUERY_STEP*256.0/255.0
# Height, central slopes; two nu; two L_Deygout; C73 direct/total extra channels.
QUERY_CHANNELS = (58, 63, 61, 66, 50, 51, 52, 53)
PAYLOAD_CHANNEL_NAMES = ("query_h_over100", "query_sx_over5", "query_sy_over5",
    "nu415", "nu5775", "Ld415", "Ld5775", "C73_direct415", "C73_direct5775",
    "C73_total415", "C73_total5775", "tx_dx_over_Rmax", "tx_dy_over_Rmax",
    "tx_radius_over_Rmax", "coordinate_support")


def registered_query_tx(published_rc):
    """[B,2] published integer (row,col) -> registered query-image (row,col).

    Coordinate conversion only; TX altitude is not used by this routing branch.
    C149/C150 native mesh: x=col*255/256, y=(row+1)*255/256.
    Query coordinate native q(j)=1/3+j*QUERY_STEP; mesh spacing=256/255 m.
    """
    if published_rc.ndim != 2 or published_rc.shape[1] != 2:
        raise ValueError("Expected published TX coordinates [B,2] in row,col order")
    p = published_rc.float()
    native = torch.stack(((p[:, 0]+1.0)*255.0/256.0,
                          p[:, 1]*255.0/256.0), 1)
    return (native-QUERY_ORIGIN)/QUERY_STEP


def tx_from_inputs(x):
    """Recover original integer TX metadata from the frozen dx/128,dy/128."""
    if x.ndim != 4 or x.shape[1:] != (CHANNELS, 256, 256):
        raise ValueError("Expected the C150 teacher-free input [B,68,256,256]")
    published = torch.stack((-128*x[:, 6, 0, 0].float(),
                             -128*x[:, 5, 0, 0].float()), 1).round()
    return registered_query_tx(published)


def _native_projection():
    q = torch.linspace(QUERY_ORIGIN, 255.0-QUERY_ORIGIN, 256,
                       dtype=torch.float64)
    y, x = torch.meshgrid(q, q, indexing="ij")
    iy, ix = y.floor().long(), x.floor().long()
    fy, fx = y-iy, x-ix
    a = iy*256+ix
    lower = fy <= fx
    # Diagonal a--d: lower a,b,d; upper a,c,d.
    indices = torch.stack((a, torch.where(lower, a+1, a+256), a+257), 0)
    weights = torch.stack((torch.where(lower, 1-fx, 1-fy),
                           torch.where(lower, fx-fy, fy-fx),
                           torch.where(lower, fy, fx)), 0)
    return indices.flatten(1), weights.float().flatten(1)


@dataclass
class Geometry:
    mode: str
    forward_grid: torch.Tensor       # [B,ntheta,nr,2], grid_sample x,y
    support: torch.Tensor            # [B,1,ntheta,nr], coordinate-only 0/1
    inverse_grid: torch.Tensor       # [B,256,256,2], on appended seam row
    coordinate_fields: torch.Tensor  # [B,3,ntheta,nr], dx/Rmax,dy/Rmax,r/Rmax
    rmax: torch.Tensor               # [B], query-image pixel units


@torch.no_grad()
def sampling_geometry(tx_query_rc, mode="polar", ntheta=256, nr=192):
    """Build label-free fixed-radius or Cartesian forward/inverse coordinates."""
    if mode not in MODES:
        raise ValueError("Unknown routing mode")
    if tx_query_rc.ndim != 2 or tx_query_rc.shape[1] != 2:
        raise ValueError("Expected registered query TX [B,2] in row,col order")
    if ntheta < 4 or nr < 2:
        raise ValueError("The sampling grid needs ntheta>=4 and nr>=2")
    tx = tx_query_rc.detach().float()
    b, device = len(tx), tx.device
    corners = tx.new_tensor(((0., 0.), (0., 255.), (255., 0.), (255., 255.)))
    rmax = ((tx[:, None]-corners[None]).square().sum(-1).amax(1)).sqrt()
    # A rectangle of nonzero area ensures Rmax>0 for every finite TX.
    if mode == "polar":
        theta = torch.arange(ntheta, device=device, dtype=torch.float32)
        theta = theta*(2.0*math.pi/ntheta)
        radius = torch.linspace(0, 1, nr, device=device)[None, None]*rmax[:, None, None]
        sx = tx[:, 1, None, None]+theta.cos()[None, :, None]*radius
        sy = tx[:, 0, None, None]+theta.sin()[None, :, None]*radius
    else:
        yy, xx = torch.meshgrid(torch.linspace(0, 255, ntheta, device=device),
                                torch.linspace(0, 255, nr, device=device), indexing="ij")
        sx, sy = xx[None].expand(b, -1, -1), yy[None].expand(b, -1, -1)
    tolerance = 8*torch.finfo(torch.float32).eps*255.0
    support = ((sx >= -tolerance) & (sx <= 255+tolerance) &
               (sy >= -tolerance) & (sy <= 255+tolerance))[:, None].float()
    # Safe clamping does not create terrain outside the coordinate support.
    forward = torch.stack((sx.clamp(0, 255)/127.5-1,
                           sy.clamp(0, 255)/127.5-1), -1)
    dx, dy = sx-tx[:, 1, None, None], sy-tx[:, 0, None, None]
    coordinate = torch.stack((dx/rmax[:, None, None], dy/rmax[:, None, None],
                              (dx.square()+dy.square()).sqrt()/rmax[:, None, None]), 1)
    yy, xx = torch.meshgrid(torch.arange(256, device=device, dtype=torch.float32),
                            torch.arange(256, device=device, dtype=torch.float32), indexing="ij")
    if mode == "polar":
        dx, dy = xx[None]-tx[:, 1, None, None], yy[None]-tx[:, 0, None, None]
        frac_r = (dx.square()+dy.square()).sqrt()/rmax[:, None, None]
        frac_theta = torch.remainder(torch.atan2(dy, dx), 2.0*math.pi)/(2.0*math.pi)
        # First theta row appended at index ntheta; align_corners denominator ntheta.
        inverse = torch.stack((frac_r*2-1, frac_theta*2-1), -1)
    else:
        # Replicate final Cartesian row; same appended shape as periodic polar.
        ix = (xx/127.5-1)[None].expand(b, -1, -1)
        iy = ((yy/255.0)*(ntheta-1)/ntheta*2-1)[None].expand(b, -1, -1)
        inverse = torch.stack((ix, iy), -1)
    return Geometry(mode, forward, support, inverse, coordinate, rmax)


def sample_query(field, geometry):
    """Sample query-image fields and remove all geometric padding."""
    sampled = F.grid_sample(field.float(), geometry.forward_grid,
                            mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled*geometry.support


def inverse_query(sampled, geometry, return_support=False):
    """Invert with bounded support normalization; uncovered query pixels are 0.

    The same fixed threshold 0.25 is used in both modes. Interpolated support
    below it is discarded; accepted normalization can amplify by at most 4.
    Coordinate support is not a label-validity input.
    """
    if sampled.shape[0] != geometry.support.shape[0] or sampled.shape[-2:] != geometry.support.shape[-2:]:
        raise ValueError("Sampled fields and geometry shape differ")
    last = slice(0, 1) if geometry.mode == "polar" else slice(-1, None)
    values = sampled.float()*geometry.support
    values = torch.cat((values, values[:, :, last, :]), 2)
    support = torch.cat((geometry.support, geometry.support[:, :, last, :]), 2)
    numerator = F.grid_sample(values, geometry.inverse_grid, mode="bilinear",
                              padding_mode="zeros", align_corners=True)
    denominator = F.grid_sample(support, geometry.inverse_grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
    accepted = denominator >= INVERSE_SUPPORT_THRESHOLD
    result = numerator/denominator.clamp_min(INVERSE_SUPPORT_THRESHOLD)
    result = result*accepted
    return (result, denominator, accepted) if return_support else result


def _rotate_half(v):
    a, b = v[..., ::2], v[..., 1::2]
    return torch.stack((-b, a), -1).flatten(-2)


class RadialBlock(nn.Module):
    """Identical bidirectional sequence mechanism for either coordinate routing."""
    def __init__(self, width=32, heads=4, attention_chunk=128):
        super().__init__()
        if width % heads or (width//heads) % 2:
            raise ValueError("Attention head dimension must be positive and even")
        self.width, self.heads = width, heads
        self.attention_chunk = int(attention_chunk)
        if self.attention_chunk < 1:
            raise ValueError("Attention chunks must be positive")
        self.norm_conv = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm_attn = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.attention_out = nn.Linear(width, width)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(),
                                nn.Linear(2*width, width))

    def forward(self, z, support):
        b, c, ntheta, nr = z.shape
        mask = support[:, 0].reshape(b*ntheta, nr)
        m = mask[:, :, None].to(z.dtype)
        seq = z.permute(0, 2, 3, 1).reshape(b*ntheta, nr, c)*m
        a = self.norm_conv(seq)*m
        a = self.pointwise(F.silu(self.depthwise(a.transpose(1, 2)))).transpose(1, 2)
        seq = (seq+a)*m
        a = self.norm_attn(seq)*m
        q, k, v = self.qkv(a).reshape(b*ntheta, nr, 3, self.heads,
                                     c//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # Parameter-free relative sequence positions, identical in both modes.
        dim = c//self.heads
        freq = torch.exp(-math.log(10000.0)*torch.arange(0, dim, 2,
                         device=z.device, dtype=torch.float32)/dim)
        phase = torch.arange(nr, device=z.device, dtype=torch.float32)[:, None]*freq[None]
        cos = phase.cos().repeat_interleave(2, -1)[None, None].to(q.dtype)
        sin = phase.sin().repeat_interleave(2, -1)[None, None].to(q.dtype)
        q, k = q*cos+_rotate_half(q)*sin, k*cos+_rotate_half(k)*sin
        valid_keys = mask.bool()
        # All-invalid rays get one zero dummy KV. Query/output support still 0.
        valid_keys = valid_keys | ((~valid_keys.any(1))[:, None] &
                                  (torch.arange(nr, device=z.device)[None] == 0))
        key_values = mask[:, None, :, None].to(v.dtype)
        k, v = k*key_values, v*key_values
        pieces = []
        for start in range(0, len(seq), self.attention_chunk):
            stop = start+self.attention_chunk
            pieces.append(F.scaled_dot_product_attention(q[start:stop], k[start:stop],
                v[start:stop], attn_mask=valid_keys[start:stop, None, None, :],
                dropout_p=0.0, is_causal=False))
        a = torch.cat(pieces, 0).transpose(1, 2).reshape_as(seq)
        seq = (seq+self.attention_out(a))*m
        seq = (seq+self.ff(self.norm_ff(seq)*m))*m
        return seq.reshape(b, ntheta, nr, c).permute(0, 3, 1, 2).contiguous()


class TxRadialBranch(nn.Module):
    """Equal-capacity active Cartesian/polar branch; mode has no learned weights."""
    def __init__(self, mode="polar", width=32, out_channels=OUT_CHANNELS,
                 ntheta=256, nr=192, depth=2, heads=4, attention_chunk=128):
        super().__init__()
        if mode not in MODES or depth not in (2, 3):
            raise ValueError("Use a known routing mode and two or three radial layers")
        self.mode, self.ntheta, self.nr = mode, int(ntheta), int(nr)
        index, weight = _native_projection()
        self.register_buffer("_projection_index", index, persistent=False)
        self.register_buffer("_projection_weight", weight, persistent=False)
        self.stem = nn.Conv2d(PAYLOAD_CHANNELS, width, 1)
        self.blocks = nn.ModuleList([RadialBlock(width, heads, attention_chunk)
                                     for _ in range(depth)])
        # Normal active initialization in both modes; do not zero this branch.
        self.head = nn.Conv2d(width, out_channels, 1)

    def query_fields(self, x):
        if x.ndim != 4 or x.shape[1:] != (CHANNELS, 256, 256):
            raise ValueError("Expected [B,68,256,256], without label/mask channels")
        native = x[:, (0, 3, 4)].float().flatten(2)
        terrain = sum(native.index_select(2, self._projection_index[j])*
                      self._projection_weight[j][None, None] for j in range(3))
        terrain = terrain.reshape(len(x), 3, 256, 256)
        return torch.cat((terrain, x[:, QUERY_CHANNELS].float()), 1).clamp(-12, 12)

    def forward(self, x, tx_query_rc=None):
        if tx_query_rc is None:
            tx_query_rc = tx_from_inputs(x)
        if len(tx_query_rc) != len(x) or tx_query_rc.device != x.device:
            raise ValueError("TX and input batch/device must match")
        geometry = sampling_geometry(tx_query_rc, self.mode, self.ntheta, self.nr)
        routed = sample_query(self.query_fields(x), geometry)
        payload = torch.cat((routed, geometry.coordinate_fields*geometry.support,
                             geometry.support), 1)
        z = self.stem(payload)
        z = z*geometry.support.to(z.dtype)
        for block in self.blocks:
            z = block(z, geometry.support)
        return inverse_query(self.head(z)*geometry.support, geometry)


def _cpu_checks():
    """Synthetic numerical/gradient tests only; never reads data or starts CUDA."""
    torch.set_num_threads(2)
    torch.manual_seed(151023)
    if torch.cuda.is_initialized():
        raise AssertionError("CPU checks require an uninitialized CUDA runtime")
    result = {"device": "cpu", "labels_read": False, "masks_read": False,
              "gpu_initialized": False, "grid": [256, 192],
              "inverse_support_threshold": INVERSE_SUPPORT_THRESHOLD}
    raw = torch.tensor(((127, 127), (0, 0), (255, 255), (0, 255), (255, 0)))
    tx = registered_query_tx(raw)
    native_recovered = tx*QUERY_STEP+QUERY_ORIGIN
    native_expected = torch.stack(((raw[:, 0]+1)*255/256, raw[:, 1]*255/256), 1)
    torch.testing.assert_close(native_recovered, native_expected, rtol=0, atol=2e-5)
    assert tx[1, 1] < 0 and tx[2, 0] > 255
    result["registered_tx_query_rc"] = tx.tolist()
    yy, xx = torch.meshgrid(torch.arange(256).float(), torch.arange(256).float(), indexing="ij")
    fields = torch.stack((torch.ones_like(xx), xx/255, yy/255), 0)[None]
    coverage = {}
    for mode in MODES:
        g = sampling_geometry(tx, mode)
        restored, den, valid = inverse_query(sample_query(fields.expand(len(tx), -1, -1, -1), g),
                                              g, return_support=True)
        constant_error = (restored[:, 0]-1).abs()[valid[:, 0]].max().item()
        assert constant_error < 2e-6
        if (~valid).any():
            assert restored.masked_select(~valid.expand_as(restored)).abs().max().item() == 0
        errors = (restored[:, 1:]-fields[:, 1:]).abs()
        good_error = errors.masked_select(valid.expand(-1, 2, -1, -1))
        per_map = []
        for i in range(len(tx)):
            accepted = valid[i].float().mean().item()
            d = den[i][valid[i]]
            per_map.append({"coverage": accepted, "min_support_all": den[i].min().item(),
                            "min_support_accepted": d.min().item(),
                            "all_invalid_sequences": int((g.support[i, 0].sum(1) == 0).sum())})
        coverage[mode] = {"maps": per_map, "constant_max_error": constant_error,
                         "linear_mean_error": good_error.mean().item(),
                         "linear_max_error": good_error.max().item()}
        assert min(a["coverage"] for a in per_map) > 0.96
        assert good_error.mean() < 0.003 and good_error.max() < 0.04
    result["roundtrip"] = coverage
    # Exact anti0 interpolation of an affine native mesh (including both triangles).
    probe = TxRadialBranch(ntheta=16, nr=24)
    x = torch.zeros(1, CHANNELS, 256, 256)
    x[:, 0] = (2*xx+3*yy+7)/100
    q = torch.linspace(QUERY_ORIGIN, 255-QUERY_ORIGIN, 256)
    qy, qx = torch.meshgrid(q, q, indexing="ij")
    expected = (2*qx+3*qy+7)/100
    torch.testing.assert_close(probe.query_fields(x)[:, 0],
        expected.clamp(-12, 12)[None], atol=2e-6, rtol=2e-7)
    # Synthetic periodic data vanish at r=0 and are continuous across theta=0.
    g = sampling_geometry(tx[:1], "polar")
    theta = torch.arange(256)*2*math.pi/256
    radius = torch.linspace(0, 1, 192)
    signal = ((theta.cos()+2*theta.sin())[:, None]*radius[None])[None, None]
    signal = signal*g.support
    restored, den, _ = inverse_query(signal, g, True)
    dx, dy = xx-tx[0, 1], yy-tx[0, 0]
    target = (dx+2*dy)/g.rmax[0]
    seam = (dx > 10) & (dy.abs() < 1.1) & (den[0, 0] >= .999)
    seam_error = (restored[0, 0]-target)[seam].abs().max().item()
    assert seam_error < 3e-4
    result["seam_max_error"] = seam_error
    # 90-degree coordinate covariance: polar grid rotates by exactly 64 theta rows.
    center = torch.tensor(((127.5, 127.5),))
    g = sampling_geometry(center, "polar")
    rotated = torch.stack((-g.forward_grid[..., 1], g.forward_grid[..., 0]), -1)
    torch.testing.assert_close(rotated, torch.roll(g.forward_grid, -64, 1),
                               atol=2e-6, rtol=0)
    result["coordinate_quarter_turn_max_error"] = (
        rotated-torch.roll(g.forward_grid, -64, 1)).abs().max().item()
    # Same weights, active outputs, finite nonzero parameter gradients in both arms.
    a = TxRadialBranch("cartesian", ntheta=16, nr=24, attention_chunk=8)
    b = TxRadialBranch("polar", ntheta=16, nr=24, attention_chunk=8)
    b.load_state_dict(a.state_dict(), strict=True)
    count = sum(p.numel() for p in a.parameters())
    assert count == sum(p.numel() for p in b.parameters()) == 20752
    result["parameters"] = count
    x = torch.randn(1, CHANNELS, 256, 256)*0.2
    x[:, 5] = (xx[None]-0)/128
    x[:, 6] = (yy[None]-0)/128
    direct = registered_query_tx(torch.tensor(((0, 0),)))
    torch.testing.assert_close(tx_from_inputs(x), direct)
    gradient = {}
    for net in (a, b):
        inp = x.clone().requires_grad_(True)
        output = net(inp, direct)
        assert output.shape == (1, 16, 256, 256) and torch.isfinite(output).all()
        assert output.abs().max() > 0
        output.square().mean().backward()
        bad = [name for name, p in net.named_parameters() if p.grad is None or
               not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
        if bad:
            raise AssertionError("Inactive or invalid gradients: "+str(bad))
        assert inp.grad is not None and torch.isfinite(inp.grad).all()
        gradient[net.mode] = {"output_abs_mean": output.abs().mean().item(),
                             "input_gradient_abs_sum": inp.grad.abs().sum().item(),
                             "all_parameter_gradients_finite_nonzero": True}
    result["active_gradient"] = gradient
    # An entirely invalid ray must remain zero and finite through both blocks.
    z = torch.randn(1, 32, 4, 12, requires_grad=True)
    support = torch.zeros(1, 1, 4, 12)
    output = a.blocks[0](z, support)
    assert torch.isfinite(output).all() and output.count_nonzero() == 0
    output.sum().backward()
    assert torch.isfinite(z.grad).all()
    result["all_invalid_ray"] = "finite_zero_output_and_gradient"
    assert not torch.cuda.is_initialized()
    result["status"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CPU synthetic geometry and gradient checks")
    args = parser.parse_args()
    if not args.check:
        parser.error("Only --check is available; this module never launches training")
    print(json.dumps(_cpu_checks(), ensure_ascii=False, indent=2))

