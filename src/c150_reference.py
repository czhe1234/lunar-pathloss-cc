"""C150 independent CPU numerical oracle; no Torch, CUDA, labels, or masks.

The Bullington calculation is the flat-curvature (Ce=0) specialization of
ITU-R P.526-16 (11/2025), section 4.5.1, equations 49--57, not delta-Bullington.
The single-edge approximation follows equation 31. The three-edge construction
is the frozen C150 classic root + at most one left/right actual obstruction;
it is not the Recommendation's empirical C/T multi-edge construction.

Primary reference:
https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.526-16-202511-I%21%21PDF-E.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Sequence
import numpy as np

LIGHT_SPEED = 299792458.0
FREQUENCIES_HZ = np.array([415e6, 5.775e9], dtype=np.float64)
WAVELENGTHS_M = LIGHT_SPEED / FREQUENCIES_HZ
SPACING_M = 256.0 / 255.0
HEIGHT_ZERO_EPS_M = 1e-10
SECONDARY_NU_EPS = 1e-9
FEATURE_NAMES = tuple(
    f"{key}_{band}" for band in ("415MHz", "5p775GHz")
    for key in ("nu_max", "t_nu_max", "positive_nu_integral", "L_Deygout3_dB",
                "L_Bullington_dB")
)
FEATURE_NAMES_12 = FEATURE_NAMES + ("FSPL_415MHz_dB", "FSPL_5p775GHz_dB")
P526_URL = "https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.526-16-202511-I%21%21PDF-E.pdf"


@dataclass(frozen=True)
class Peak:
    nu: float
    position_m: float
    terrain_m: float
    highest_excess_m: float


def knife_edge_loss(nu: float) -> float:
    """P.526 eq31 approximation, cutoff -.78; stable asinh equivalent."""
    nu = float(nu)
    if math.isnan(nu):
        raise ValueError("A diffraction parameter cannot be NaN")
    if nu <= -.78:
        return 0.0
    return max(0.0, 6.9 + (20.0 / math.log(10.0)) * math.asinh(nu - .1))


def _profile(distances_m: Sequence[float], heights_m: Sequence[float],
             max_step_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(distances_m, dtype=np.float64)
    z = np.asarray(heights_m, dtype=np.float64)
    if (d.ndim != 1 or z.shape != d.shape or len(d) < 2
            or not np.isfinite(d).all() or not np.isfinite(z).all()
            or not np.all(np.diff(d) > 0)):
        raise ValueError("Finite strictly increasing distances and matching heights required")
    if not math.isfinite(max_step_m) or max_step_m <= 0:
        raise ValueError("max_step_m must be positive")
    d = d - d[0]
    # Preserve every supplied terrain corner and add an independent <=1m grid.
    n = max(2, int(math.ceil(float(d[-1]) / max_step_m)))
    points = np.unique(np.concatenate((d, np.linspace(0., float(d[-1]), n + 1))))
    return points, np.interp(points, d, z)


def _pieces(d: np.ndarray, z: np.ndarray, lo: float, hi: float,
            tx_z: float, rx_z: float):
    nodes = np.concatenate(([lo], d[(d > lo) & (d < hi)], [hi]))
    heights = np.interp(nodes, d, z)
    fraction = (nodes - lo) / (hi - lo)
    for left, right, hleft, hright in zip(fraction[:-1], fraction[1:],
                                         heights[:-1], heights[1:]):
        # A root reconstructed from t can lie one ulp from a terrain knot;
        # normalized fractions may then coincide exactly. Such a zero interval
        # has no contribution to the continuous profile or its integral.
        if right <= left:
            continue
        terrain_slope = (hright - hleft) / (right - left)
        a = hleft - terrain_slope * left - tx_z
        b = terrain_slope - (rx_z - tx_z)
        yield float(left), float(right), float(a), float(b)


def nu_peak(distances_m: Sequence[float], heights_m: Sequence[float],
            tx_z_m: float, rx_z_m: float, wavelength_m: float,
            lo_m: float | None = None, hi_m: float | None = None) -> Peak:
    """Analytic supremum over a continuous piecewise-linear terrain profile.

    This function intentionally exposes infinite source-buried limits. The
    feature interface below rejects invalid terminal clearance instead of
    inventing a finite cap. Sub-path endpoints on an edge have limiting nu=0.
    """
    d = np.asarray(distances_m, dtype=np.float64)
    z = np.asarray(heights_m, dtype=np.float64)
    lo = float(d[0] if lo_m is None else lo_m)
    hi = float(d[-1] if hi_m is None else hi_m)
    if hi <= lo or wavelength_m <= 0:
        raise ValueError("Positive sub-path length and wavelength required")
    initial = float(np.interp(lo, d, z)) - float(tx_z_m)
    final = float(np.interp(hi, d, z)) - float(rx_z_m)
    if abs(initial) <= HEIGHT_ZERO_EPS_M:
        initial = 0.0
    if abs(final) <= HEIGHT_ZERO_EPS_M:
        final = 0.0
    if initial > 0:
        return Peak(math.inf, lo, float(np.interp(lo, d, z)), initial)
    if final > 0:
        return Peak(math.inf, hi, float(np.interp(hi, d, z)), final)
    best = 0.0 if initial == 0 or final == 0 else -math.inf
    at = lo if initial == 0 else hi if final == 0 else lo
    highest = max(initial, final)
    scale = math.sqrt(2.0 / (wavelength_m * (hi - lo)))
    for left, right, a, b in _pieces(d, z, lo, hi, float(tx_z_m), float(rx_z_m)):
        candidates = [left, right]
        denominator = b + 2.0 * a
        if abs(denominator) > 1e-20:
            stationary = a / denominator
            if left < stationary < right:
                candidates.append(stationary)
        for t in candidates:
            excess = a + b * t
            if abs(excess) <= HEIGHT_ZERO_EPS_M:
                excess = 0.0
            highest = max(highest, excess)
            if 0 < t < 1:
                value = excess * scale / math.sqrt(t * (1.0 - t))
                if value > best:
                    best = value
                    at = lo + t * (hi - lo)
    return Peak(float(best), float(at), float(np.interp(at, d, z)), float(highest))


def positive_nu_integral(distances_m: Sequence[float], heights_m: Sequence[float],
                         tx_z_m: float, rx_z_m: float, wavelength_m: float) -> float:
    """Exact integral over normalized root-path t, splitting terrain and e=0."""
    d = np.asarray(distances_m, dtype=np.float64)
    z = np.asarray(heights_m, dtype=np.float64)
    length = float(d[-1] - d[0])
    scale = math.sqrt(2.0 / (float(wavelength_m) * length))

    def primitive(t: float, a: float, b: float) -> float:
        t = min(1.0, max(0.0, t))
        return ((2*a+b)*math.asin(math.sqrt(t))
                - b*math.sqrt(max(0.0, t*(1.0-t))))

    answer = 0.0
    for left, right, a, b in _pieces(d, z, float(d[0]), float(d[-1]),
                                    float(tx_z_m), float(rx_z_m)):
        cuts = [left, right]
        if b != 0:
            zero = -a / b
            if left < zero < right:
                cuts.insert(1, zero)
        for x, y in zip(cuts[:-1], cuts[1:]):
            if a + b * ((x+y)/2) > 0:
                answer += scale * (primitive(y, a, b) - primitive(x, a, b))
    # Tiny negative roundoff must not turn a nonnegative feature negative.
    return max(0.0, float(answer))


def deygout_three_edge(distances_m: Sequence[float], heights_m: Sequence[float],
                       tx_z_m: float, rx_z_m: float, wavelength_m: float,
                       root: Peak | None = None) -> dict:
    """Frozen classical three-edge approximation with a root Fresnel term.

    Root uses J(nu) for nu>-.78, including a negative grazing root. Additional
    left/right terms require a genuinely positive sub-path nu>1e-9; endpoints
    tending to nu=0 never create artificial ~6dB secondary edges.
    """
    d = np.asarray(distances_m, dtype=np.float64)
    z = np.asarray(heights_m, dtype=np.float64)
    main = root or nu_peak(d, z, tx_z_m, rx_z_m, wavelength_m)
    root_loss = knife_edge_loss(main.nu)
    losses = {"root": root_loss, "left": 0.0, "right": 0.0}
    peaks = {"root": main.nu, "left": None, "right": None}
    positions = {"root": main.position_m, "left": None, "right": None}
    if main.nu > -.78 and d[0] < main.position_m < d[-1] and math.isfinite(main.nu):
        left = nu_peak(d, z, tx_z_m, main.terrain_m, wavelength_m,
                       float(d[0]), main.position_m)
        right = nu_peak(d, z, main.terrain_m, rx_z_m, wavelength_m,
                        main.position_m, float(d[-1]))
        for name, peak in (("left", left), ("right", right)):
            peaks[name] = peak.nu
            positions[name] = peak.position_m
            if peak.nu > SECONDARY_NU_EPS:
                losses[name] = knife_edge_loss(peak.nu)
    return {"loss_dB": float(sum(losses.values())), "loss_terms_dB": losses,
            "nu": peaks, "positions_m": positions,
            "model": "classic_three_edge_with_root_Fresnel_term"}


def bullington(distances_m: Sequence[float], heights_m: Sequence[float],
               tx_z_m: float, rx_z_m: float, wavelength_m: float,
               root: Peak | None = None) -> dict:
    """P.526-16 section4.5.1 eq49--57, Ce=0, using explicit km units."""
    d_m = np.asarray(distances_m, dtype=np.float64)
    z = np.asarray(heights_m, dtype=np.float64)
    d_km = (d_m - d_m[0]) / 1000.0
    total_km = float(d_km[-1])
    if total_km <= 0 or wavelength_m <= 0:
        raise ValueError("Positive path length and wavelength required")
    main = root or nu_peak(d_m, z, tx_z_m, rx_z_m, wavelength_m)
    interior = (d_km > 0) & (d_km < total_km)
    di, hi = d_km[interior], z[interior]
    if len(di):
        slope_tx = float(np.max((hi-tx_z_m)/di))       # Eq49, Ce=0
        slope_rx = float(np.max((hi-rx_z_m)/(total_km-di)))  # Eq53
    else:
        slope_tx = slope_rx = -math.inf
    direct_slope = (rx_z_m-tx_z_m)/total_km           # Eq50
    # Exactly grazing profiles have a 0/0 construction; use its LoS limit.
    is_los = slope_tx < direct_slope or main.highest_excess_m <= HEIGHT_ZERO_EPS_M
    point_km = None
    if is_los:
        nu = main.nu                                  # Eq51, continuous terrain
    else:
        denominator = slope_tx+slope_rx
        if not math.isfinite(denominator) or denominator <= 0:
            raise ArithmeticError("Invalid trans-horizon Bullington slopes")
        point_km = (rx_z_m-tx_z_m+slope_rx*total_km)/denominator  # Eq54
        if not 0 < point_km < total_km:
            raise ArithmeticError("Bullington point outside path")
        line = (tx_z_m*(total_km-point_km)+rx_z_m*point_km)/total_km
        excess = tx_z_m+slope_tx*point_km-line
        nu = excess*math.sqrt(.002*total_km /
                              (wavelength_m*point_km*(total_km-point_km)))  # Eq55
    uncorrected = knife_edge_loss(nu)                 # Eq52/Eq56
    correction = -math.expm1(-uncorrected/6.0)*(10.0+.02*total_km)  # Eq57
    return {"loss_dB": float(uncorrected+correction),
            "knife_edge_loss_dB": uncorrected, "nu_b": float(nu),
            "correction_dB": float(correction), "distance_km": total_km,
            "slope_tx_m_per_km": slope_tx, "slope_rx_m_per_km": slope_rx,
            "bullington_distance_km": point_km, "LoS_case": bool(is_los),
            "model": "P526_16_section4_5_1_Ce0_only"}


def profile_features(distances_m: Sequence[float], heights_m: Sequence[float],
                     tx_z_m: float, rx_z_m: float, *,
                     cache_feature_rounding: bool = False,
                     details: bool = False):
    """Return (10 band-major geometry/loss features, two registered FSPL values).

    Optional C149 cache compatibility rounds ONLY reported nu(lambda=1m)/t
    features. The actual root position used by Deygout always stays Float64.
    """
    d, z = _profile(distances_m, heights_m)
    if (not math.isfinite(tx_z_m) or not math.isfinite(rx_z_m)
            or tx_z_m <= z[0] or rx_z_m <= z[-1]):
        raise ValueError("C150 requires both terminals strictly above reconstructed terrain")
    root1 = nu_peak(d, z, tx_z_m, rx_z_m, 1.0)
    if not math.isfinite(root1.nu):
        raise ValueError("Invalid infinite profile parameter")
    t_main = (root1.position_m-d[0])/(d[-1]-d[0])
    out = []
    diagnostics = []
    for wavelength in WAVELENGTHS_M:
        root = Peak(root1.nu/math.sqrt(wavelength), root1.position_m,
                    root1.terrain_m, root1.highest_excess_m)
        deygout = deygout_three_edge(d, z, tx_z_m, rx_z_m, wavelength, root)
        bull = bullington(d, z, tx_z_m, rx_z_m, wavelength, root)
        integral = positive_nu_integral(d, z, tx_z_m, rx_z_m, wavelength)
        report_nu = root.nu
        report_t = t_main
        if cache_feature_rounding:
            report_nu = float(np.float32(float(np.float32(root1.nu))/math.sqrt(wavelength)))
            report_t = float(np.float32(t_main))
        out += [report_nu, report_t, integral, deygout["loss_dB"], bull["loss_dB"]]
        diagnostics.append({"root": root.__dict__, "deygout": deygout, "bullington": bull})
    d3d = math.hypot(float(d[-1]), float(rx_z_m-tx_z_m))
    fspl = 20*np.log10(4*math.pi*max(d3d,1e-9)*FREQUENCIES_HZ/LIGHT_SPEED)
    values = np.asarray(out,np.float64)
    if not np.isfinite(values).all() or not np.isfinite(fspl).all():
        raise ArithmeticError("Nonfinite physical features")
    if details:
        return values, fspl, {"bands": diagnostics, "positions_m": d,
                             "terrain_m": z, "max_step_m": float(np.diff(d).max()),
                             "d2d_m": float(d[-1]), "d3d_m": d3d}
    return values, fspl


def triangle_height(h: np.ndarray, x: float, y: float) -> float:
    """Independent explicit anti=0 triangle interpolation."""
    ix = min(254,max(0,int(math.floor(x))))
    iy = min(254,max(0,int(math.floor(y))))
    fx, fy = x-ix, y-iy
    a,b,c,d = map(float,(h[iy,ix],h[iy,ix+1],h[iy+1,ix],h[iy+1,ix+1]))
    if fy > fx:
        return (1-fy)*a+(fy-fx)*c+fx*d
    return (1-fx)*a+(fx-fy)*b+fy*d


def registered_profile(h: np.ndarray, tr: int, tc: int, r: int, c: int):
    """Enumerate all mesh intersections; independent of the CUDA DDA traversal."""
    h = np.asarray(h,dtype=np.float64)
    if h.shape != (256,256) or not np.isfinite(h).all():
        raise ValueError("A finite 256x256 heightmap is required")
    if not all(0 <= int(v) <= 255 for v in (tr,tc,r,c)):
        raise ValueError("Pixel coordinates must be within0..255")
    tx_x, tx_y = tc*255/256,(tr+1)*255/256
    rx_x = 1/3+c*(255-2/3)/255
    rx_y = 1/3+r*(255-2/3)/255
    tx_z = math.floor(float(h[min(tr+1,255),tc]))+3
    rx_z = triangle_height(h,rx_x,rx_y)+1
    dx,dy = rx_x-tx_x,rx_y-tx_y
    distance = SPACING_M*math.hypot(dx,dy)
    if distance <= 0:
        raise ValueError("Horizontal path length must be positive")
    fractions = [0.,1.]
    for origin,direction,cuts in ((tx_x,dx,range(1,255)),
                                  (tx_y,dy,range(1,255)),
                                  (tx_x-tx_y,dx-dy,range(-254,255))):
        if abs(direction)>1e-16:
            fractions += [(k-origin)/direction for k in cuts
                          if 0 < (k-origin)/direction < 1]
    fractions += np.linspace(0.,1.,max(2,int(math.ceil(distance)))+1).tolist()
    fractions = np.unique(fractions)
    positions = fractions*distance
    heights = np.array([triangle_height(h,tx_x+dx*t,tx_y+dy*t)
                        for t in fractions],np.float64)
    return positions, heights, float(tx_z), float(rx_z)


def reference_pixel(h: np.ndarray,tr: int,tc: int,r: int,c: int,*,
                    cache_feature_rounding: bool = True,details: bool = False):
    d,z,tx,rx = registered_profile(h,tr,tc,r,c)
    return profile_features(d,z,tx,rx,cache_feature_rounding=cache_feature_rounding,
                            details=details)


def reference_array(h: np.ndarray,tr: int,tc: int,r: int,c: int,*,
                    cache_feature_rounding: bool = True) -> np.ndarray:
    values,fspl = reference_pixel(h,tr,tc,r,c,
                                 cache_feature_rounding=cache_feature_rounding)
    return np.concatenate((values,fspl))


def self_check() -> dict:
    fixtures = []
    assert knife_edge_loss(-.78)==0 and knife_edge_loss(-.779)>0
    assert abs(knife_edge_loss(0)-6.032852208563606)<1e-12
    cases = [
        ("flat_grazing",[0,300],[0,0],3.,1.),
        ("isolated_nonbinary_ridge",[0,37.123456789,100],[0,10,0],3.,1.),
        ("two_obstacles",[0,30,70,100],[0,8,11,0],3.,1.),
        ("three_obstacles",[0,20,50,80,100],[0,7,12,8,0],3.,1.),
        ("tangent",[0,50,100],[0,2,0],3.,1.),
        ("short_path",[0,.005],[0,0],3.,1.),
        ("near_equal_edges",[0,30,70,100],[0,8,7.200001,0],3.,1.),
    ]
    for name,d,z,tx,rx in cases:
        f,fs,detail = profile_features(d,z,tx,rx,details=True)
        dd = np.asarray(d,float)
        reverse,reverse_fs = profile_features(dd[-1]-dd[::-1],np.asarray(z)[::-1],
                                              rx,tx)
        # Path reversal preserves scalar diffraction features and3D FSPL.
        same = [0,2,3,4,5,7,8,9]
        assert np.allclose(f[same],reverse[same],atol=3e-9,rtol=1e-10),(name,f,reverse)
        assert np.allclose(fs,reverse_fs,atol=1e-12,rtol=0)
        assert detail["max_step_m"]<=1+1e-12
        assert np.all(f[[3,4,8,9]]>=0) and np.all(f[[2,7]]>=0)
        assert abs((fs[1]-fs[0])-20*math.log10(FREQUENCIES_HZ[1]/FREQUENCIES_HZ[0]))<1e-12
        if name=="flat_grazing":
            expected=-np.sqrt(8*tx*rx/(WAVELENGTHS_M*dd[-1]))
            assert np.allclose(f[[0,5]],expected,atol=1e-12,rtol=0)
            assert np.allclose(f[[1,6]],tx/(tx+rx),atol=1e-12)
            assert f[3]>0 and f[8]==0
        if name in ("isolated_nonbinary_ridge","flat_grazing","tangent"):
            for b in detail["bands"]:
                assert b["deygout"]["loss_terms_dB"]["left"]==0
                assert b["deygout"]["loss_terms_dB"]["right"]==0
                assert abs(b["deygout"]["loss_dB"]-knife_edge_loss(b["root"]["nu"]))<1e-12
        if name=="three_obstacles":
            assert all(v>0 for v in detail["bands"][0]["deygout"]["loss_terms_dB"].values())
        # Independently integrate midpoint samples for the positive root feature.
        t=(np.arange(200000,dtype=float)+.5)/200000
        ground=np.interp(t*dd[-1],dd,z)
        e=ground-(tx*(1-t)+rx*t)
        mid=np.maximum(e*np.sqrt(2/(WAVELENGTHS_M[0]*dd[-1]*t*(1-t))),0).mean()
        assert abs(mid-f[2])<3e-7,(name,mid,f[2])
        fixtures.append({"name":name,"features":f.tolist(),"FSPL_dB":fs.tolist(),
                         "positive_integral_midpoint_abs_error":float(abs(mid-f[2]))})
    rejected=[]
    for name,tx,rx in (("buried_source",-1.,1.),("touching_source",0.,1.),
                       ("buried_receiver",3.,-1.)):
        try:profile_features([0,100],[0,0],tx,rx)
        except ValueError:rejected.append(name)
        else:raise AssertionError("Invalid endpoint was not rejected")
    try:profile_features([0,0],[0,0],3.,1.)
    except ValueError:rejected.append("zero_length")
    else:raise AssertionError("Zero-length path was not rejected")
    raw_buried=nu_peak([0,100],[0,0],-1.,1.,WAVELENGTHS_M[0])
    assert math.isinf(raw_buried.nu)
    # A dyadic affine terrain has an independently known clearance extremum.
    h=(np.arange(256)[None,:]/32+np.arange(256)[:,None]/64).astype(np.float32)
    d,z,tx,rx=registered_profile(h,70,60,193,201)
    f,fs=profile_features(d,z,tx,rx)
    clearance=tx-triangle_height(h,60*255/256,71*255/256)
    expected=-np.sqrt(8*clearance/(WAVELENGTHS_M*d[-1]))
    assert np.allclose(f[[0,5]],expected,atol=2e-12,rtol=0)
    assert np.allclose(f[[1,6]],clearance/(clearance+1),atol=2e-12,rtol=0)
    return {"status":"CPU_oracle_checks_passed","fixtures":fixtures,
            "invalid_cases_rejected":rejected,
            "registered_affine_profile_analytic_check":True,
            "feature_names":list(FEATURE_NAMES_12),"official_reference":P526_URL,
            "GPU_used":False,"Torch_imported":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if not args.check:
        parser.error("Use --check; the reference module never generatesfull feature caches")
    print(json.dumps(self_check(),ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
