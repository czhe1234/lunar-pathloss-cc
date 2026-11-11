"""C143 native CUDA lattice check on the frozen C141 map1819 scene."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np

import c142_backend as fixed


ROOT = Path(__file__).resolve().parents[1]
MAP = 1819
TARGET_PIXEL = 5926
C141 = ROOT / "audit/c141_direct_receiver"
INPUT = fixed.INPUTS / str(MAP)
OUT = ROOT / "audit/c143_case00414/native_cuda"
WAVELENGTH_GAIN = np.square(299792458.0 / (4 * np.pi * np.array([415e6, 5.775e9])))


def pathloss(power):
    with np.errstate(divide="ignore"):
        return -10 * np.log10(power)


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_summary(result, manifest_digest, output_dir):
    fit = next(rec for rec in json.loads((C141 / "fit_panel.json").read_text())["records"]
               if rec["map"] == MAP)
    frozen_record = next(rec for rec in json.loads((fixed.INPUTS / "manifest.json").read_text())["records"]
                         if rec["map"] == MAP)
    with np.load(C141 / f"fit_replay/{MAP}.npz") as cpu_file:
        cpu = {key: cpu_file[key] for key in cpu_file.files}
    with np.load(fixed.AUDIT / f"cuda/{MAP}.npz") as matched_file:
        matched = {key: matched_file[key] for key in matched_file.files}
    for name, data in (("matched_cuda", matched), ("native_cuda", result)):
        assert str(data["input_manifest_sha256"]) == manifest_digest, name
        for key in ("pixels", "ids", "weights"):
            assert np.array_equal(data[key], cpu[key]), (name, key)
    with np.load(ROOT / f"audit/c136_native/cache/{MAP}.npz") as cache:
        inv = cache["inv"][fit["cache_positions"]]
        assert np.array_equal(cache["uf"][inv], cpu["ids"])
        dw = cache["dw"][inv].astype(float)
    weights = cpu["weights"]
    active_donor = weights > 1e-12
    assert np.all((dw > 0) | (weights[..., None] < 1e-7))
    area = cpu["area4_native"][..., None]
    extra = np.maximum(np.divide(area * (1-dw), dw, out=np.zeros_like(dw), where=dw > 0), 0)
    labels = np.asarray(fit["labels"], dtype=float)
    assert np.array_equal(cpu["pixels"], fit["pixels"])
    groups = {
        "target": cpu["pixels"] == TARGET_PIXEL,
        "ordinary": np.isin(cpu["pixels"], frozen_record["ordinary_pixels"]),
        "remaining_cases_on_map": np.isin(cpu["pixels"], frozen_record["case_pixels"]),
    }
    assert groups["target"].sum() == 1 and groups["ordinary"].sum() == 8
    summaries = {}
    predictions = {}
    for name, data in (("archived_llvm", cpu), ("matched_cuda", matched), ("native_cuda", result)):
        direct = data["native_full_native"].astype(float)
        assert np.isfinite(direct).all() and np.all(direct >= 0)
        gain = direct[..., None] * WAVELENGTH_GAIN
        total_gain = (direct[..., None] + extra) * WAVELENGTH_GAIN
        direct_linear = np.sum(gain * weights[..., None], axis=1)
        total_linear = np.sum(total_gain * weights[..., None], axis=1)
        donor_direct_pl = pathloss(gain)
        total_log = np.sum(np.where(active_donor[..., None], pathloss(total_gain), 0)
                           * weights[..., None], axis=1)
        weakest = pathloss(np.min(np.where(active_donor[..., None], gain, np.inf), axis=1))
        pred = {
            "direct_linear_PL_dB": pathloss(direct_linear),
            "direct_weakest_donor_PL_dB": weakest,
            "total_linear_PL_dB": pathloss(total_linear),
            "total_log_PL_dB": total_log,
        }
        predictions[name] = pred
        group_results = {}
        for group, selected in groups.items():
            record = {"points": int(selected.sum())}
            for key, values in pred.items():
                finite = np.isfinite(values[selected])
                error = np.where(finite, values[selected], labels[selected])-labels[selected]
                n = finite.sum(0)
                record[key] = {
                    "values": values[selected] if group == "target" else None,
                    "RMSE_dB": np.sqrt(np.divide((error*error).sum(0), n,
                                                out=np.full(2, np.nan), where=n > 0)),
                    "finite_points": n,
                }
            conflict = labels[selected] - weakest[selected]
            record["weakest_direct_excess_above_label_dB"] = conflict if group == "target" else None
            record["weakest_direct_conflicts_above_1dB"] = np.sum(conflict > 1, axis=0)
            group_results[group] = record
        summaries[name] = group_results
    comparisons = {}
    native = result["native_full_native"].astype(float)
    for name, data in (("archived_llvm", cpu), ("matched_cuda", matched)):
        baseline = data["native_full_native"].astype(float)
        common = active_donor & (baseline > 0) & (native > 0)
        delta = pathloss(native[common])-pathloss(baseline[common])
        comparisons[name] = {
            "positive_weight_common_donor_entries": int(common.sum()),
            "donor_support_changes": int(np.count_nonzero(active_donor & ((baseline > 0) != (native > 0)))),
            "native_minus_reference_direct_donor_PL_dB_q50_q90_q99_max_abs":
                np.quantile(np.abs(delta), [.5, .9, .99, 1]),
            "target_native_minus_reference_total_linear_PL_dB":
                predictions["native_cuda"]["total_linear_PL_dB"][groups["target"]]
                - predictions[name]["total_linear_PL_dB"][groups["target"]],
        }
    summary = {
        "scope": "Fixed TRAIN map1819, sample00414_00, frozen C141 query donors and eight ordinary controls",
        "purpose": (
            "Exact candidate-mesh 3m AGL TX with native CUDA rays and original receiver stepping"
            if str(result.get("condition", "c141_original_tx")) == "candidate_exact_3m_agl_same_xy"
            else "Native CUDA lattice arithmetic versus identical-world-vector C142 CUDA and archived LLVM"
        ),
        "condition": str(result.get("condition", "c141_original_tx")),
        "tx": np.asarray(result.get("tx", cpu["tx"])),
        "reference_tx": cpu["tx"],
        "ray_id_sha256": str(result.get("ray_id_sha256", "")),
        "ray_id_hash_encoding": "SHA256 of consecutive little-endian uint32 global direction IDs, without file header",
        "agl_source_sha256": str(result.get("agl_source_sha256", "")),
        "background_condition": "Original C136 background is frozen. Total-power scores are diagnostic; only TX, direct rays and receiver stepping change.",
        "frequency_hz": [415e6, 5.775e9],
        "input_manifest_sha256": manifest_digest,
        "gpu_uuid": str(result["gpu_uuid"]),
        "ray_count": int(result["selected_rays"]),
        "native_original_spawn_function": str(result["spawn_function"]),
        "labels_used_to_form_rays_or_power": False,
        "target_labels_dB": labels[groups["target"]],
        "methods": summaries,
        "native_comparisons": comparisons,
    }
    fixed.write_json(output_dir / "comparison.json", clean(summary))
    print(json.dumps(clean({"event": "native_comparison", "summary": str(output_dir / "comparison.json"),
                            "native_comparisons": comparisons})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-agl", action="store_true",
                        help="Use the saved candidate-mesh surface+3m TX at the same XY")
    args = parser.parse_args()
    output_dir = OUT.with_name("native_cuda_agl3") if args.exact_agl else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = INPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = fixed.file_hash(manifest_path)
    for name, expected in manifest["sha256"].items():
        assert fixed.file_hash(INPUT / name) == expected, name
    assert fixed.file_hash(C141 / f"fit_replay/{MAP}.npz") == manifest["reference_npz_sha256"]
    with np.load(INPUT / "queries.npz") as file:
        queries = {key: file[key] for key in file.files}
    tx = queries["tx"].copy()
    condition = "c141_original_tx"
    agl_digest = ""
    if args.exact_agl:
        agl_path = ROOT / "audit/c143_case00414/agl_control/area256.npz"
        agl_digest = fixed.file_hash(agl_path)
        with np.load(agl_path) as agl:
            assert np.array_equal(agl["old_tx"], tx)
            tx = agl["corrected_tx"].copy()
            assert np.array_equal(tx[:2], queries["tx"][:2])
            assert abs(float(tx[2]-agl["surface_height"])-3.) < 1e-12
        with np.load(C141 / f"fit_replay/{MAP}.npz") as cpu:
            vertices = cpu["native_vertices"].astype(float)
        rays = fixed.direction_indices(vertices-tx)
        condition = "candidate_exact_3m_agl_same_xy"
    else:
        rays = np.load(INPUT / "ids.npy", mmap_mode="r")
    assert rays.dtype == np.uint32 and np.all(rays[1:] > rays[:-1])
    ray_digest = hashlib.sha256(np.ascontiguousarray(rays, dtype="<u4").tobytes()).hexdigest()
    dest = output_dir / f"{MAP}.npz"
    if dest.exists():
        with np.load(dest) as file:
            result = {key: file[key] for key in file.files}
        assert str(result["input_manifest_sha256"]) == digest
        assert str(result["variant"]) == "cuda_ad_mono_polarized"
        assert int(result["selected_rays"]) == len(rays)
        if args.exact_agl:
            assert str(result["condition"]) == condition
            assert np.array_equal(result["tx"], tx)
            assert str(result["ray_id_sha256"]) == ray_digest
            assert str(result["agl_source_sha256"]) == agl_digest
        assert str(result["solver_sha256"]) == fixed.file_hash(
            ROOT / ".rt_packages/sionna/rt/radio_map_solvers/radio_map_solver.py")
        write_summary(result, digest, output_dir)
        return
    mi, dr, rt, device = fixed.rt_environment("cuda")
    scene, rx = fixed.load_scene(mi, rt, INPUT, tx)
    module = importlib.import_module("sionna.rt.radio_map_solvers.radio_map_solver")
    original_lattice = module.fibonacci_lattice
    original_spawn = module.spawn_ray_from_sources
    assert original_spawn.__module__ == "sionna.rt.utils.ray_tracing"
    spawn_name = original_spawn.__module__ + "." + original_spawn.__name__
    solver = rt.RadioMapSolver()
    power = np.zeros(rx.face_count(), dtype=np.float64)
    start = time.monotonic()
    try:
        for offset in range(0, len(rays), fixed.BATCH):
            batch = rays[offset:offset+fixed.BATCH]

            def native_lattice(size):
                assert size == len(batch)
                return fixed.lattice(mi, dr, batch)

            module.fibonacci_lattice = native_lattice
            assert module.spawn_ray_from_sources is original_spawn
            radio = solver(scene, measurement_surface=rx, samples_per_tx=len(batch),
                           max_depth=0, los=True, specular_reflection=False,
                           diffuse_reflection=False, refraction=False, seed=13)
            power += radio.path_gain.numpy()[0].astype(float) * (len(batch)/fixed.N) * fixed.POWER_SCALE
    finally:
        module.fibonacci_lattice = original_lattice
    assert np.isfinite(power).all() and np.all(power >= 0)
    result = {
        "pixels": queries["pixels"], "ids": queries["ids"], "weights": queries["weights"],
        "native_full_native": power[queries["ids"]],
        "tx": tx, "reference_tx": queries["tx"], "condition": condition,
        "ray_id_sha256": ray_digest, "agl_source_sha256": agl_digest,
        "selected_rays": len(rays), "global_N": fixed.N, "batch": fixed.BATCH, "seed": 13,
        "variant": mi.variant(), "gpu_uuid": device, "sionna_rt": rt.__version__,
        "mitsuba": mi.__version__, "drjit": dr.__version__,
        "input_manifest_sha256": digest, "spawn_function": spawn_name,
        "solver_sha256": fixed.file_hash(ROOT / ".rt_packages/sionna/rt/radio_map_solvers/radio_map_solver.py"),
        "elapsed_seconds": time.monotonic()-start,
    }
    temp = dest.with_suffix(".partial.npz")
    np.savez_compressed(temp, **result)
    temp.replace(dest)
    write_summary(result, digest, output_dir)


if __name__ == "__main__":
    main()
