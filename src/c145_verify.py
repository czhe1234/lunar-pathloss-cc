"""Independently score saved C145 Float32 predictions against original TRAIN arrays.

This script imports NumPy only. It never imports the GPU training modules.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    "all",
    "C82_abs_error_ge_5dB",
    "C82_underestimated_loss_ge_5dB",
    "C82_overestimated_loss_ge_5dB",
    "C82_abs_error_lt_5dB",
    "C82_abs_error_le_1dB",
)
STAT_NAMES = (
    "sse", "baseline_sse", "count", "error_sum", "absolute_error_sum",
    "baseline_error_sum", "correction_sum", "absolute_correction_sum",
    "correction_aligned", "correction_square", "gross_sse_removed",
    "gross_sse_added", "improved_pixels", "worsened_pixels",
    "correct_correction_direction",
)
BOOTSTRAP_SEED = 145073
BOOTSTRAP_REPLICATES = 2000


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def panel_identity(model_digest, ids):
    ordered_ids = np.asarray(ids, dtype="<i8")
    return {
        "model_sha256": model_digest,
        "panel_ids_sha256": hashlib.sha256(ordered_ids.tobytes()).hexdigest(),
        "shape": [len(ids), 2, 256, 256],
        "prediction_dtype": "float32",
        "forward_precision": "BF16 autocast",
    }


def validate_export(run, tag, panels, paths):
    """Check completed export identities and hash each saved prediction once."""
    export_path = run/f"{tag}_export.json"
    export_identity_before = file_identity(export_path)
    export_bytes = export_path.read_bytes()
    exported = json.loads(export_bytes)
    if exported.get("tag") != tag:
        raise ValueError(f"Completed export tag differs: {export_path}")
    model_digest = exported.get("model_sha256")
    if (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(character not in "0123456789abcdef" for character in model_digest)
    ):
        raise ValueError(f"Invalid exported model SHA256: {export_path}")
    for key, expected in (("prediction_dtype", "float32"), ("forward_precision", "BF16 autocast")):
        if exported.get(key) != expected:
            raise ValueError(f"Completed export {key} differs: {export_path}")
    exported_panels = exported.get("panels")
    if not isinstance(exported_panels, dict):
        raise ValueError(f"Completed export panel records are missing: {export_path}")
    validated = {}
    for name, ids in panels.items():
        if name not in paths:
            continue
        path = paths[name]
        sidecar = path.with_suffix(".json")
        sidecar_identity_before = file_identity(sidecar)
        sidecar_bytes = sidecar.read_bytes()
        recorded = json.loads(sidecar_bytes)
        completed = exported_panels.get(name)
        if not isinstance(completed, dict) or completed.get("file") != path.name:
            raise ValueError(f"Completed export prediction filename differs for {name}")
        if {key: value for key, value in completed.items() if key != "file"} != recorded:
            raise ValueError(f"Completed export and prediction sidecar disagree for {name}")
        expected = panel_identity(model_digest, ids)
        for key, value in expected.items():
            if recorded.get(key) != value:
                raise ValueError(f"Prediction sidecar {key} differs for {name}")
        before = file_identity(path)
        if recorded.get("file_size_bytes") != before[2]:
            raise ValueError(f"Prediction file size differs for {name}")
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(prediction.shape) != expected["shape"] or prediction.dtype != np.float32:
            raise ValueError(f"Prediction file layout differs for {name}")
        del prediction
        prediction_digest = file_hash(path)
        after = file_identity(path)
        if before != after:
            raise RuntimeError(f"Prediction changed during export verification: {path}")
        if recorded.get("prediction_sha256") != prediction_digest:
            raise ValueError(f"Prediction file SHA256 differs for {name}")
        if file_identity(sidecar) != sidecar_identity_before:
            raise RuntimeError(f"Prediction sidecar changed during export verification: {sidecar}")
        validated[name] = {
            "file_identity": after,
            "prediction_sha256": prediction_digest,
            "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "panel_ids_sha256": expected["panel_ids_sha256"],
        }
    if set(validated) != set(paths):
        raise ValueError("Completed export does not contain all required prediction panels")
    if file_identity(export_path) != export_identity_before:
        raise RuntimeError(f"Completed export record changed during verification: {export_path}")
    return {
        "model_sha256": model_digest,
        "checkpoint": exported.get("checkpoint"),
        "export_record_sha256": hashlib.sha256(export_bytes).hexdigest(),
        "prediction_dtype": "float32",
        "forward_precision": "BF16 autocast",
        "panels": validated,
    }


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(np.asarray(numerator, float), np.nan),
                     where=np.asarray(denominator) > 0)


def group_statistic(prediction, baseline, labels, selected):
    """Return per-map, per-band sufficient statistics on one fixed group."""
    error = prediction-labels
    base_error = baseline-labels
    correction = prediction-baseline
    removed = np.square(base_error)-np.square(error)

    def total(values):
        return np.where(selected, values, 0).sum((2, 3), dtype=np.float64)

    return {
        "sse": total(np.square(error)),
        "baseline_sse": total(np.square(base_error)),
        "count": selected.sum((2, 3), dtype=np.int64),
        "error_sum": total(error),
        "absolute_error_sum": total(np.abs(error)),
        "baseline_error_sum": total(base_error),
        "correction_sum": total(correction),
        "absolute_correction_sum": total(np.abs(correction)),
        "correction_aligned": total(2*(labels-baseline)*correction),
        "correction_square": total(np.square(correction)),
        "gross_sse_removed": total(np.maximum(removed, 0)),
        "gross_sse_added": total(np.maximum(-removed, 0)),
        "improved_pixels": total(removed > 0),
        "worsened_pixels": total(removed < 0),
        "correct_correction_direction": total(correction*(labels-baseline) > 0),
    }


def summarize(statistics):
    value = {name: np.asarray(statistics[name], float) for name in STAT_NAMES}
    count = value["count"]
    removed = value["baseline_sse"]-value["sse"]
    assert np.allclose(
        removed, value["gross_sse_removed"]-value["gross_sse_added"], rtol=1e-9, atol=1e-6
    )
    assert np.allclose(
        removed, value["correction_aligned"]-value["correction_square"], rtol=1e-9, atol=1e-6
    )
    return {
        "RMSE_dB": np.sqrt(divide(value["sse"], count)),
        "C82_RMSE_dB": np.sqrt(divide(value["baseline_sse"], count)),
        "SSE_dB2": value["sse"],
        "C82_SSE_dB2": value["baseline_sse"],
        "valid_pixels": count.astype(np.int64),
        "SSE_removed_dB2": removed,
        "SSE_ratio_to_C82": divide(value["sse"], value["baseline_sse"]),
        "mean_error_dB": divide(value["error_sum"], count),
        "C82_mean_error_dB": divide(value["baseline_error_sum"], count),
        "MAE_dB": divide(value["absolute_error_sum"], count),
        "mean_correction_dB": divide(value["correction_sum"], count),
        "mean_absolute_correction_dB": divide(value["absolute_correction_sum"], count),
        "alignment_2r_delta_dB2": value["correction_aligned"],
        "correction_square_cost_dB2": value["correction_square"],
        "gross_SSE_removed_dB2": value["gross_sse_removed"],
        "gross_SSE_added_dB2": value["gross_sse_added"],
        "improved_pixels": value["improved_pixels"].astype(np.int64),
        "worsened_pixels": value["worsened_pixels"].astype(np.int64),
        "correct_correction_direction_pixels": value["correct_correction_direction"].astype(np.int64),
        "correct_correction_direction_fraction": divide(value["correct_correction_direction"], count),
    }


def paired_terrain_bootstrap(sse, base_sse, count):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(len(sse), size=(BOOTSTRAP_REPLICATES, len(sse)))
    denominator = count[samples].sum(1)
    candidate = np.sqrt(divide(sse[samples].sum(1), denominator))
    baseline = np.sqrt(divide(base_sse[samples].sum(1), denominator))
    return {
        "unit": "terrain, retaining all maps/pixels for each sampled terrain",
        "terrains": len(sse),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "gain_C82_minus_candidate_dB_95CI": np.quantile(baseline-candidate, [.025, .975], axis=0).T,
        "method_limit": "Repeatedly used development monitor and checkpoint selection share this panel.",
    }


def check_fixture():
    labels = np.array([[[[0., 10.], [10., 4.]], [[1., 9.], [7., 2.]]]])
    baseline = np.array([[[[0., 4.], [17., 5.]], [[1., 3.], [14., 3.]]]])
    valid = np.ones_like(labels, dtype=bool)
    valid[0, 1, 1, 1] = False
    stat = group_statistic(baseline, baseline, labels, valid)
    summary = summarize({key: value.sum(0) for key, value in stat.items()})
    assert np.array_equal(summary["SSE_removed_dB2"], [0., 0.])
    assert np.array_equal(summary["valid_pixels"], [4, 3])
    assert np.array_equal(summary["SSE_dB2"], [86., 85.])
    corrected = group_statistic(labels, baseline, labels, valid)
    summary = summarize({key: value.sum(0) for key, value in corrected.items()})
    assert np.array_equal(summary["SSE_dB2"], [0., 0.])
    assert np.array_equal(summary["SSE_removed_dB2"], [86., 85.])


def score_panel(name, ids, prediction_path, raw_labels, raw_masks, baseline, lookup, rows, batch,
                export_record):
    prediction_identity_before = file_identity(prediction_path)
    if prediction_identity_before != export_record["file_identity"]:
        raise RuntimeError(f"Prediction changed after completed-export verification: {prediction_path}")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (len(ids), 2, 256, 256) or prediction.dtype != np.float32:
        raise ValueError(f"Wrong saved prediction shape/dtype: {prediction_path}")
    assert len(ids) == len(set(ids))
    map_statistics = {key: np.zeros((len(GROUPS), len(ids), 2)) for key in STAT_NAMES}
    new_errors = {"C82_le_1_to_error_gt_3": np.zeros(2, np.int64),
                  "C82_le_1_to_error_gt_5": np.zeros(2, np.int64),
                  "C82_ge_5_to_error_lt_1": np.zeros(2, np.int64)}
    for start in range(0, len(ids), batch):
        selected_ids = ids[start:start+batch]
        labels = np.stack([array[selected_ids] for array in raw_labels], 1).astype(np.float64)
        valid = np.stack([
            np.unpackbits(array[selected_ids], axis=1, bitorder="big").reshape(len(selected_ids), 256, 256)
            for array in raw_masks
        ], 1).astype(bool)
        base = np.asarray(baseline[[lookup[index] for index in selected_ids]], dtype=np.float64)
        estimate = np.asarray(prediction[start:start+len(selected_ids)], dtype=np.float64)
        if not np.isfinite(estimate).all():
            raise ValueError(f"Nonfinite saved predictions in {prediction_path}, map offset {start}")
        residual = labels-base
        masks = (
            valid,
            valid & (np.abs(residual) >= 5),
            valid & (residual >= 5),
            valid & (residual <= -5),
            valid & (np.abs(residual) < 5),
            valid & (np.abs(residual) <= 1),
        )
        for group_index, group_mask in enumerate(masks):
            statistics = group_statistic(estimate, base, labels, group_mask)
            for key, values in statistics.items():
                map_statistics[key][group_index, start:start+len(selected_ids)] = values
        error = np.abs(estimate-labels)
        good = valid & (np.abs(residual) <= 1)
        new_errors["C82_le_1_to_error_gt_3"] += (good & (error > 3)).sum((0, 2, 3))
        new_errors["C82_le_1_to_error_gt_5"] += (good & (error > 5)).sum((0, 2, 3))
        new_errors["C82_ge_5_to_error_lt_1"] += (masks[1] & (error < 1)).sum((0, 2, 3))
    totals = {key: value.sum(1) for key, value in map_statistics.items()}
    for key in ("sse", "baseline_sse", "count"):
        assert np.allclose(totals[key][1]+totals[key][4], totals[key][0], rtol=1e-12, atol=1e-6)
        assert np.allclose(totals[key][2]+totals[key][3], totals[key][1], rtol=1e-12, atol=1e-6)
    terrains = sorted({rows[index]["terrain_id"] for index in ids})
    terrain_lookup = {terrain: i for i, terrain in enumerate(terrains)}
    terrain_index = np.array([terrain_lookup[rows[index]["terrain_id"]] for index in ids])
    terrain_sse = np.zeros((len(terrains), 2))
    terrain_base_sse = np.zeros_like(terrain_sse)
    terrain_count = np.zeros_like(terrain_sse)
    np.add.at(terrain_sse, terrain_index, map_statistics["sse"][0])
    np.add.at(terrain_base_sse, terrain_index, map_statistics["baseline_sse"][0])
    np.add.at(terrain_count, terrain_index, map_statistics["count"][0])
    per_map_rmse = np.sqrt(divide(map_statistics["sse"][0], map_statistics["count"][0]))
    prediction_identity_after = file_identity(prediction_path)
    if prediction_identity_before != prediction_identity_after:
        raise RuntimeError(f"Prediction changed during independent scoring: {prediction_path}")
    summary = {
        "maps": len(ids), "terrains": len(terrains),
        "prediction_path": str(prediction_path),
        "prediction_sha256": export_record["prediction_sha256"],
        "prediction_sidecar_sha256": export_record["sidecar_sha256"],
        "ordered_panel_ids_sha256": export_record["panel_ids_sha256"],
        "groups": {
            group: summarize({key: values[i] for key, values in totals.items()})
            for i, group in enumerate(GROUPS)
        },
        "new_errors": new_errors,
        "worst_map_RMSE_dB": np.nanmax(per_map_rmse, axis=0),
        "maps_both_below_1dB": int(np.count_nonzero(np.all(per_map_rmse < 1, axis=1))),
    }
    arrays = {
        f"{name}_map_ids": np.asarray(ids, np.int64),
        f"{name}_map_terrain_index": terrain_index,
        f"{name}_terrain_names": np.asarray(terrains),
        f"{name}_terrain_sse": terrain_sse,
        f"{name}_terrain_C82_sse": terrain_base_sse,
        f"{name}_terrain_count": terrain_count,
        **{f"{name}_map_group_{key}": values for key, values in map_statistics.items()},
    }
    return summary, arrays, (terrain_sse, terrain_base_sse, terrain_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/c145_capacity_native"))
    parser.add_argument("--tag", choices=("selected", "fit_best", "last"), default="selected")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    check_fixture()
    run = args.run if args.run.is_absolute() else ROOT/args.run
    config_path = run/"config.json"
    config = json.loads(config_path.read_text())
    panels = config["panels"]
    frozen = json.loads((ROOT/"audit/c105_context/protocol.json").read_text())
    paths = {name: run/f"{args.tag}_{name}_prediction.npy" for name in ("fit", "monitor")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing final predictions: "+"; ".join(missing))
    for name in ("fit", "monitor"):
        if set(panels[name]) != set(frozen[name]) or len(panels[name]) != len(frozen[name]):
            raise ValueError(f"C145 {name} IDs differ from the fixed C105 panel")
    export_verified = validate_export(run, args.tag, panels, paths)
    data = ROOT.parent/"data"
    with (data/"train_index.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fit_terrains = {rows[index]["terrain_id"] for index in panels["fit"]}
    monitor_terrains = {rows[index]["terrain_id"] for index in panels["monitor"]}
    if fit_terrains & monitor_terrains:
        raise ValueError("FIT and monitor terrains overlap")
    labels = [np.load(data/f"train_pl{band}.npy", mmap_mode="r") for band in ("415", "58")]
    masks = [np.load(data/f"train_mask{band}.npy", mmap_mode="r") for band in ("415", "58")]
    fit_base_path = ROOT/"audit/c101_residual/base0of1/prediction.npy"
    monitor_base_path = ROOT/"audit/c88_mesh/base.npy"
    fit_index_path = ROOT/"audit/c101_residual/base0of1/indices.json"
    monitor_index_path = ROOT/"audit/c88_mesh/base_index.json"
    fit_ids = json.loads(fit_index_path.read_text())
    monitor_ids = json.loads(monitor_index_path.read_text())["ids"]
    sources = {
        "fit": (np.load(fit_base_path, mmap_mode="r"), {int(i): j for j, i in enumerate(fit_ids)}),
        "monitor": (np.load(monitor_base_path, mmap_mode="r"), {int(i): j for j, i in enumerate(monitor_ids)}),
    }
    result = {
        "status": "verified",
        "tag": args.tag,
        "scope": "Fixed TRAIN-internal full-image FIT and repeatedly used terrain-disjoint monitor; original validity masks",
        "frequency_hz": [415_000_000, 5_775_000_000],
        "large_error_definition": "abs(original TRAIN label - frozen development C82) >= 5 dB",
        "error_sign_definition": "error=prediction-label; positive correction increases predicted path loss",
        "source": "Saved Float32 predictions, original TRAIN Float16 labels and bit-packed masks, NumPy Float64 arithmetic",
        "official_VAL_used": False,
        "verification_source_sha256": file_hash(Path(__file__)),
        "config_sha256": file_hash(config_path),
        "export_identity": {
            key: value for key, value in export_verified.items() if key != "panels"
        },
        "frozen_protocol_sha256": file_hash(ROOT/"audit/c105_context/protocol.json"),
        "baseline_fit_index_sha256": file_hash(fit_index_path),
        "baseline_monitor_index_sha256": file_hash(monitor_index_path),
        "panels": {},
    }
    statistics = {"group_names": np.asarray(GROUPS)}
    for name in ("fit", "monitor"):
        baseline, lookup = sources[name]
        summary, arrays, terrain_stats = score_panel(name, panels[name], paths[name], labels, masks,
                                                     baseline, lookup, rows, args.batch,
                                                     export_verified["panels"][name])
        result["panels"][name] = summary
        statistics.update(arrays)
        if name == "monitor":
            full = summary["groups"]["all"]
            assert full["valid_pixels"].tolist() == [40864973, 37837148]
            assert np.allclose(full["C82_RMSE_dB"], [2.1463905073492664, 2.4180144849255294],
                               rtol=0, atol=2e-6)
            if args.tag == "selected":
                summary["paired_terrain_bootstrap"] = paired_terrain_bootstrap(*terrain_stats)
        print(json.dumps(clean({
            "panel": name,
            "all": summary["groups"]["all"],
            "large_error": summary["groups"]["C82_abs_error_ge_5dB"],
            "new_errors": summary["new_errors"],
        })), flush=True)
    destination = run/f"{args.tag}_verified.json"
    temporary = run/f"{args.tag}_terrain_sse.partial.npz"
    np.savez_compressed(temporary, **statistics)
    temporary.replace(run/f"{args.tag}_terrain_sse.npz")
    atomic_json(destination, result)
    print(json.dumps({"status": "verified", "result": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
