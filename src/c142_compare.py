"""Compare C142 receiver trajectories and frozen FIT power checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/c142_backend"
C141 = ROOT / "audit/c141_direct_receiver"
FREQUENCIES = np.array([415e6, 5.775e9])
WAVELENGTH_GAIN = np.square(299792458.0 / (4 * np.pi * FREQUENCIES))
MARGIN_DB = 1.0


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


def pathloss(power):
    with np.errstate(divide="ignore"):
        return -10 * np.log10(power)


def summarize(values):
    values = np.asarray(values, float)
    return {
        "n": values.size,
        "q50_q90_q99_max": np.quantile(values, [.5, .9, .99, 1]) if values.size else [],
    }


def trace_data(data):
    ray_ids = data["trace_ids"]
    order = np.argsort(ray_ids)
    if len(np.unique(ray_ids)) != len(ray_ids):
        raise ValueError("Trace ray IDs must be unique")
    kind = data["trace_kind"][:, order]
    face = data["trace_face"][:, order]
    coordinates = {key: data[f"trace_{key}"][:, order] for key in ("p", "o", "d")}
    if (
        kind.shape != face.shape
        or not np.isin(kind, (0, 1, 2, 3)).all()
        or any(v.shape != kind.shape + (3,) for v in coordinates.values())
    ):
        raise ValueError("Invalid trajectory event arrays")
    directions = data["directions"][order]
    cosine = data["trace_cos"][:, order]
    if directions.shape != (len(ray_ids), 3) or cosine.shape != kind.shape:
        raise ValueError("Invalid launch direction or incidence cosine arrays")
    return ray_ids[order], directions, kind, face, coordinates, cosine


def compare_traces(llvm, cuda):
    a_ids, a_dir, a_kind, a_face, a_xyz, a_cos = trace_data(llvm)
    b_ids, b_dir, b_kind, b_face, b_xyz, b_cos = trace_data(cuda)
    if not np.array_equal(a_ids, b_ids) or not np.array_equal(a_dir, b_dir):
        raise ValueError("LLVM and CUDA trace launch inputs differ")
    changes = dict(receiver_order=0, terminal_event=0, ground_hit_status=0,
                   first_ground_face_among_both=0, both_ground_hits=0,
                   llvm_unfinished=0, cuda_unfinished=0)
    deltas = dict(rx_intersection=[], rx_advance_origin=[], rx_advance_direction=[],
                  rx_incidence_cosine=[], ground_intersection_among_both=[])
    examples = []
    for ray, ray_id in enumerate(a_ids):
        ai = np.flatnonzero(a_kind[:, ray])
        bi = np.flatnonzero(b_kind[:, ray])
        ak, bk = a_kind[ai, ray], b_kind[bi, ray]
        arx, brx = ai[ak == 2], bi[bk == 2]
        ag, bg = ai[ak == 3], bi[bk == 3]
        a_terminal = int(ak[-1]) if ak.size and ak[-1] in (1, 3) else 0
        b_terminal = int(bk[-1]) if bk.size and bk[-1] in (1, 3) else 0
        a_rx_faces, b_rx_faces = a_face[arx, ray], b_face[brx, ray]
        a_ground = int(a_face[ag[0], ray]) if ag.size else None
        b_ground = int(b_face[bg[0], ray]) if bg.size else None
        different_rx = not np.array_equal(a_rx_faces, b_rx_faces)
        changes["receiver_order"] += different_rx
        changes["terminal_event"] += a_terminal != b_terminal
        changes["ground_hit_status"] += bool(ag.size) != bool(bg.size)
        changes["llvm_unfinished"] += not a_terminal
        changes["cuda_unfinished"] += not b_terminal
        if ag.size and bg.size:
            changes["both_ground_hits"] += 1
            changes["first_ground_face_among_both"] += a_ground != b_ground
            deltas["ground_intersection_among_both"].append(
                np.linalg.norm(a_xyz["p"][ag[0], ray] - b_xyz["p"][bg[0], ray])
            )
        for left, right in zip(arx, brx):
            if a_face[left, ray] != b_face[right, ray]:
                continue
            deltas["rx_incidence_cosine"].append(abs(float(a_cos[left, ray]) - b_cos[right, ray]))
            for key, label in (
                ("p", "rx_intersection"), ("o", "rx_advance_origin"),
                ("d", "rx_advance_direction"),
            ):
                deltas[label].append(
                    np.linalg.norm(a_xyz[key][left, ray] - b_xyz[key][right, ray])
                )
        if len(examples) < 8 and (different_rx or a_terminal != b_terminal or a_ground != b_ground):
            examples.append({
                "ray_id": ray_id,
                "llvm_rx_faces": a_rx_faces,
                "cuda_rx_faces": b_rx_faces,
                "llvm_terminal_event": a_terminal,
                "cuda_terminal_event": b_terminal,
                "llvm_first_ground_face": a_ground,
                "cuda_first_ground_face": b_ground,
            })
    return {
        "trace_rays": len(a_ids),
        "changed": changes,
        "coordinate_deltas": {
            key: {
                **summarize(np.asarray(values)[np.isfinite(values)]),
                "nonfinite_comparisons": np.count_nonzero(~np.isfinite(values)),
                "unit": "dimensionless" if key.endswith(("direction", "cosine")) else "m",
            } for key, values in deltas.items()
        },
        "first_changed_trajectories": examples,
    }


def aligned_donors(data, reference):
    power = data["native_full_native"].astype(float)
    for key in ("pixels", "ids", "weights"):
        if not np.array_equal(data[key], reference[key]):
            raise ValueError(f"Replayed {key} differ from the C141 cache")
    if power.shape != reference["ids"].shape or not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("Invalid replay donor power")
    return power


def map_power(donors, weights):
    positive = weights[..., None] > 1e-12
    gain = donors * WAVELENGTH_GAIN
    log_gain = np.where(positive, pathloss(gain), 0)
    return {
        "power_interpolation": np.sum(weights[..., None] * gain, axis=1),
        "db_interpolation": np.power(10.0, -np.sum(weights[..., None] * log_gain, axis=1) / 10),
        "weakest_donor": np.min(np.where(positive, gain, np.inf), axis=1),
    }


def paired_metrics(before, after, labels, selected):
    a = before[selected]
    b = after[selected]
    y = labels[selected]
    common = np.isfinite(a) & np.isfinite(b)

    def metrics(pred, include):
        error = np.where(include, pred, y) - y
        count = np.sum(include, axis=0)
        mse = np.divide(np.sum(np.square(error), axis=0), count,
                        out=np.full(2, np.nan), where=count > 0)
        mae = np.divide(np.sum(np.abs(error), axis=0), count,
                        out=np.full(2, np.nan), where=count > 0)
        return {"RMSE_dB": np.sqrt(mse), "MAE_dB": mae, "n": count}

    return {
        "points": len(y),
        "llvm_common_support": metrics(a, common),
        "cuda_common_support": metrics(b, common),
        "llvm_finite_support": metrics(a, np.isfinite(a)),
        "cuda_finite_support": metrics(b, np.isfinite(b)),
        "llvm_nonfinite": np.sum(~np.isfinite(a), axis=0),
        "cuda_nonfinite": np.sum(~np.isfinite(b), axis=0),
    }


def compare_bound(a, b, y):
    allowed = np.power(10.0, -y / 10)
    margin = np.power(10.0, MARGIN_DB / 10)
    a_bad = a > allowed * margin
    b_bad = b > allowed * margin
    a_supported = a > 0
    b_supported = b > 0
    return {
        "points": len(y),
        "llvm_finite": np.sum(a_supported, axis=0),
        "cuda_finite": np.sum(b_supported, axis=0),
        "llvm_positive_to_cuda_zero": np.sum(a_supported & ~b_supported, axis=0),
        "llvm_zero_to_cuda_positive": np.sum(~a_supported & b_supported, axis=0),
        "llvm_over_1dB": np.sum(a_bad, axis=0),
        "cuda_over_1dB": np.sum(b_bad, axis=0),
        "llvm_over_1dB_both": np.sum(np.all(a_bad, axis=1)),
        "cuda_over_1dB_both": np.sum(np.all(b_bad, axis=1)),
        "cuda_over_1dB_either": np.sum(np.any(b_bad, axis=1)),
        "llvm_conflicts_resolved_with_finite_cuda_power": np.sum(a_bad & ~b_bad & b_supported, axis=0),
        "llvm_conflicts_with_zero_cuda_power": np.sum(a_bad & ~b_supported, axis=0),
    }


def check_input_manifest(data, path, expected_backend, reference):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if str(data["input_manifest_sha256"]) != digest:
        raise ValueError(f"Replay input manifest hash differs for {path.parent.name}")
    if str(data["variant"]) != f"{expected_backend}_ad_mono_polarized":
        raise ValueError(f"Unexpected replay variant for {path.parent.name}")
    if int(data["selected_rays"]) != int(reference["selected_rays"]):
        raise ValueError(f"Replayed ray count differs for {path.parent.name}")
    return digest


def compare_map(record, fit, work):
    map_id = record["map"]
    cpu_path = work / "llvm" / f"{map_id}.npz"
    gpu_path = work / "cuda" / f"{map_id}.npz"
    manifest_path = work / "inputs" / str(map_id) / "manifest.json"
    with np.load(C141 / "fit_replay" / f"{map_id}.npz") as ref, np.load(gpu_path) as gpu:
        digest = check_input_manifest(gpu, manifest_path, "cuda", ref)
        pixels = ref["pixels"]
        weights = ref["weights"]
        base = ref["native_full_native"].astype(float)
        alt = aligned_donors(gpu, ref)
        with np.load(ROOT / f"audit/c136_native/cache/{map_id}.npz") as cache:
            inv = cache["inv"][fit["cache_positions"]]
            if not np.array_equal(cache["uf"][inv], ref["ids"]):
                raise ValueError(f"C136 and C141 donor IDs differ for map {map_id}")
            dw = cache["dw"][inv].astype(float)
        if not np.array_equal(pixels, fit["pixels"]):
            raise ValueError(f"FIT scoring pixels differ for map {map_id}")
        if not np.all((dw > 0) | (weights[..., None] < 1e-7)):
            raise ValueError(f"C136 background cannot be recovered for map {map_id}")
        area = ref["area4_native"][..., None]
        extra = np.maximum(np.divide(area * (1 - dw), dw, out=np.zeros_like(dw), where=dw > 0), 0)

        def predictions(donor_power):
            direct = map_power(donor_power[..., None], weights)
            total = map_power(donor_power[..., None] + extra, weights)
            return direct, {key: pathloss(total[key]) for key in ("power_interpolation", "db_interpolation")}

        a_direct, a_total = predictions(base)
        b_direct, b_total = predictions(alt)
        trace = None
        parity = None
        if cpu_path.is_file():
            with np.load(cpu_path) as cpu:
                check_input_manifest(cpu, manifest_path, "llvm", ref)
                if "native_full_native" in cpu:
                    fresh = aligned_donors(cpu, ref)
                    both = (fresh > 0) & (base > 0) & (weights > 1e-12)
                    parity = {
                        "donor_support_disagreements": np.count_nonzero(
                            (fresh > 0) != (base > 0)
                        ),
                        "positive_weight_common_donor_absolute_shift_dB":
                            summarize(np.abs(pathloss(fresh[both]) - pathloss(base[both]))),
                    }
                trace = compare_traces(cpu, gpu)
        positive = weights > 1e-12
        common = positive & (base > 0) & (alt > 0)
        summary = {
            "map": map_id,
            "frozen_pixels": len(pixels),
            "input_manifest_sha256": digest,
            "cuda_uuid": str(gpu["gpu_uuid"]),
            "global_direction_ids_replayed": int(gpu["selected_rays"]),
            "cuda_native_launch_direction_differences": int(gpu["native_direction_differences"]),
            "cuda_native_launch_max_absolute_difference": float(gpu["native_direction_max_abs"]),
            "archived_llvm_positive_weight_donors": np.count_nonzero(positive & (base > 0)),
            "cuda_positive_weight_donors": np.count_nonzero(positive & (alt > 0)),
            "llvm_positive_to_cuda_zero_donors": np.count_nonzero(positive & (base > 0) & (alt == 0)),
            "llvm_zero_to_cuda_positive_donors": np.count_nonzero(positive & (base == 0) & (alt > 0)),
            "common_positive_weight_donor_absolute_shift_dB":
                summarize(np.abs(pathloss(alt[common]) - pathloss(base[common]))),
            "fresh_llvm_vs_archived_donors": parity,
            "paired_trace": trace,
        }

        frozen = {kind: set(record[f"{kind}_pixels"]) for kind in ("case", "ordinary")}
        labels = np.asarray(fit["labels"], float)
        if not frozen["case"] or not frozen["ordinary"] or frozen["case"] & frozen["ordinary"]:
            raise ValueError(f"Invalid frozen case/control selection for map {map_id}")
        selected = np.isin(pixels, list(frozen["case"] | frozen["ordinary"]))
        is_case = np.isin(pixels, list(frozen["case"]))
        if np.count_nonzero(selected) != len(frozen["case"]) + len(frozen["ordinary"]):
            raise ValueError(f"Unknown frozen case/control pixel for map {map_id}")
        original_case = np.asarray(fit["is_case"], bool)
        if np.any(selected & (is_case != original_case)):
            raise ValueError(f"Frozen case/control labels differ from C141 for map {map_id}")

        cases = []
        for i in np.flatnonzero(is_case):
            a = pathloss(a_direct["weakest_donor"][i])
            b = pathloss(b_direct["weakest_donor"][i])
            if not np.all(np.isfinite(a) & (labels[i] - a > MARGIN_DB)):
                raise ValueError(f"Archived C141 weakest donor does not violate both bands at {map_id}/{pixels[i]}")
            cases.append({
                "map": map_id,
                "pixel": int(pixels[i]),
                "row": int(pixels[i] // 256),
                "col": int(pixels[i] % 256),
                "sample_id": fit["sample_id"],
                "group": fit["group"],
                "label_dB": labels[i],
                "C82_dB": fit["C82"][i],
                "donor_ids": ref["ids"][i],
                "donor_weights": weights[i],
                "llvm_donor_normalized_power": base[i],
                "cuda_donor_normalized_power": alt[i],
                "llvm_weakest_direct_dB": a,
                "cuda_weakest_direct_dB": b,
                "cuda_weakest_direct_support": np.isfinite(b),
                "cuda_minus_llvm_weakest_direct_dB": b - a,
                "llvm_attenuation_needed_to_1dB_margin_dB": np.maximum(labels[i] - a - MARGIN_DB, 0),
                "cuda_attenuation_needed_to_1dB_margin_dB": np.where(
                    np.isfinite(b), np.maximum(labels[i] - b - MARGIN_DB, 0), np.nan
                ),
                "llvm_power_interpolation_total_dB": a_total["power_interpolation"][i],
                "cuda_power_interpolation_total_dB": b_total["power_interpolation"][i],
                "llvm_db_interpolation_total_dB": a_total["db_interpolation"][i],
                "cuda_db_interpolation_total_dB": b_total["db_interpolation"][i],
            })
        data = {
            "labels": labels[selected],
            "C82": np.asarray(fit["C82"], float)[selected],
            "is_case": is_case[selected],
            "llvm_direct": {key: value[selected] for key, value in a_direct.items()},
            "cuda_direct": {key: value[selected] for key, value in b_direct.items()},
            "llvm_total": {key: value[selected] for key, value in a_total.items()},
            "cuda_total": {key: value[selected] for key, value in b_total.items()},
        }
        return summary, cases, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.audit / "inputs" / "manifest.json").read_text())["records"]
    frozen_fit = json.loads((C141 / "fit_panel.json").read_text())["records"]
    fit_by_map = {record["map"]: record for record in frozen_fit}
    summaries, cases, data, missing = [], [], [], []
    for record in manifest:
        map_id = record["map"]
        if not (args.audit / "cuda" / f"{map_id}.npz").is_file():
            missing.append(map_id)
            continue
        summary, map_cases, values = compare_map(record, fit_by_map[map_id], args.audit)
        summaries.append(summary)
        cases.extend(map_cases)
        data.append(values)
    if not data:
        raise SystemExit("No CUDA replay output exists for the frozen C142 maps")
    labels = np.concatenate([d["labels"] for d in data])
    reference_c82 = np.concatenate([d["C82"] for d in data])
    is_case = np.concatenate([d["is_case"] for d in data])

    def combined(backend, kind, method):
        return np.concatenate([d[f"{backend}_{kind}"][method] for d in data])

    groups = {}
    for name, selected in (("remaining_counterexamples", is_case), ("ordinary_controls", ~is_case)):
        group = {"points": np.count_nonzero(selected), "direct_bounds": {}, "total_power": {}}
        for method in ("power_interpolation", "db_interpolation", "weakest_donor"):
            a = combined("llvm", "direct", method)
            b = combined("cuda", "direct", method)
            group["direct_bounds"][method] = compare_bound(a[selected], b[selected], labels[selected])
            if method == "weakest_donor":
                continue
            a = combined("llvm", "total", method)
            b = combined("cuda", "total", method)
            group["total_power"][method] = paired_metrics(a, b, labels, selected)
            common = np.isfinite(a) & np.isfinite(b)
            group["total_power"][method]["C82_common_support"] = paired_metrics(
                np.where(common, reference_c82, np.nan),
                np.where(common, reference_c82, np.nan), labels, selected
            )["llvm_common_support"]
        groups[name] = group
    if not missing and (len(cases) != 22 or np.count_nonzero(~is_case) != 80):
        raise ValueError("The full C142 panel must contain 22 counterexamples and 80 ordinary controls")
    traces = [summary["paired_trace"] for summary in summaries if summary["paired_trace"] is not None]
    totals = {
        key: sum(trace["changed"][key] for trace in traces)
        for key in traces[0]["changed"]
    } if traces else None
    output = {
        "scope": "frozen TRAIN FIT points selected for diagnosis; no model fit or validation score",
        "power_reference": "archived C141 LLVM replay; paired traces are produced in C142",
        "freq_hz": FREQUENCIES,
        "direct_conflict_margin_dB": MARGIN_DB,
        "complete_cuda_panel": not missing,
        "maps_missing_cuda": missing,
        "maps_missing_paired_trace": [s["map"] for s in summaries if s["paired_trace"] is None],
        "paired_trace_rays": sum(trace["trace_rays"] for trace in traces),
        "paired_trace_changes": totals,
        "groups": groups,
        "map_evidence": summaries,
        "case_evidence": cases,
    }
    destination = args.out or args.audit / ("compare_partial.json" if missing else "compare.json")
    temporary = destination.with_suffix(".partial")
    temporary.write_text(json.dumps(clean(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps(clean({
        "result": str(destination),
        "maps_scored": len(summaries),
        "case_points": len(cases),
        "ordinary_points": np.count_nonzero(~is_case),
        "weakest_donor": groups["remaining_counterexamples"]["direct_bounds"]["weakest_donor"],
    }), ensure_ascii=False))


if __name__ == "__main__":
    main()
