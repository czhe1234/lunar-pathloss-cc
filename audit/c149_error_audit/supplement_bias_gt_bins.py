"""C149 supplement: FP64 CPU statistics of existing saved predictions/caches only.

No model imports, geometry execution, training, inference, weight changes, or
SwanLab records. Frozen C149 files are read only; this script writes only its
explicit supplement JSON. Run from CC with CUDA_VISIBLE_DEVICES=''.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"audit/c149_error_audit"
RUN = ROOT/"runs/c145_capacity_native"
sys.path.insert(0, str(ROOT))
from src.c145_verify import validate_export

BANDS = ("415MHz", "5p775GHz")
EXPECTED_RMSE = np.array((2.14122101082334, 2.40823647956265))
EXPECTED_COUNT = np.array((40864973, 37837148), np.int64)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def file_stat(path):
    value = Path(path).stat()
    return {"size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def record(path, hashed=True):
    path = Path(path)
    return {"path": os.path.relpath(path, ROOT), **file_stat(path),
            **({"sha256": sha(path)} if hashed else {})}


class Stats:
    def __init__(self):
        self.count = np.zeros(2, np.int64)
        self.sse = np.zeros(2, np.float64)
        self.error_sum = np.zeros(2, np.float64)

    def add(self, error, squared, mask):
        self.count += mask.sum((1, 2), dtype=np.int64)
        self.sse += np.where(mask, squared, 0).sum((1, 2), dtype=np.float64)
        self.error_sum += np.where(mask, error, 0).sum((1, 2), dtype=np.float64)

    def value(self, all_stats):
        def divide(a, b):
            return np.divide(a, b, out=np.full(2, np.nan), where=b > 0)
        return {"valid_pixels": self.count.tolist(),
                "valid_pixel_fraction": divide(self.count, all_stats.count).tolist(),
                "valid_pixel_percent": (100*divide(self.count, all_stats.count)).tolist(),
                "MeanBias_dB": divide(self.error_sum, self.count).tolist(),
                "RMSE_dB": np.sqrt(divide(self.sse, self.count)).tolist(),
                "error_sum_dB": self.error_sum.tolist(), "SSE_dB2": self.sse.tolist(),
                "share_of_total_SSE_fraction": divide(self.sse, all_stats.sse).tolist(),
                "share_of_total_SSE_percent": (100*divide(self.sse, all_stats.sse)).tolist()}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This supplement requires CUDA_VISIBLE_DEVICES=''")
    started = time.monotonic()
    summary_path, protocol_path = OUT/"summary.json", OUT/"protocol.json"
    frozen = {str(p): sha(p) for p in (summary_path, protocol_path,
                                     ROOT/"src/c149_error_audit.py", ROOT/"src/c149_profiles.cu")}
    summary, protocol = read(summary_path), read(protocol_path)
    assert summary["status"] == "complete"
    ids = protocol["ordered_panel_ids"]
    assert ids == read(ROOT/"audit/c145_capacity/protocol.json")["panels"]["monitor"]
    assert len(ids) == len(set(ids)) == 628
    assert protocol["output_geometry_layout"][0] == "highest_m"
    pred_path = RUN/"selected_monitor_prediction.npy"
    identity = validate_export(RUN, "selected", {"monitor": ids}, {"monitor": pred_path})
    assert identity["model_sha256"] == summary["model_sha256"]
    assert identity["panels"]["monitor"]["prediction_sha256"] == protocol["prediction_source"]["sha256"]
    assert sha(protocol_path) == summary["protocol_sha256"]
    profiles_path = OUT/"profiles.npy"
    assert sha(profiles_path) == summary["profile_cache_sha256"]
    prediction = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    profiles = np.load(profiles_path, mmap_mode="r", allow_pickle=False)
    assert profiles.shape == (628, 5, 256, 256) and profiles.dtype == np.float32
    assert prediction.shape == (628, 2, 256, 256) and prediction.dtype == np.float32

    data = ROOT.parent/"data"
    raw_audit_path = ROOT/"audit/raw_audit.json"
    raw_audit = read(raw_audit_path)
    raw_sources = {}
    for name in ("train_index.csv", "train_pl415.npy", "train_pl58.npy",
                 "train_mask415.npy", "train_mask58.npy"):
        observed = file_stat(data/name)
        expected = raw_audit["files"][name]
        assert observed["size_bytes"] == expected["size_bytes"]
        assert observed["mtime_ns"] == expected["mtime_ns"]
        raw_sources[name] = {**record(data/name, hashed=False),
                            "frozen_raw_audit_sha256": expected["sha256"],
                            "whole_file_hash_recomputed_now": False,
                            "current_size_mtime_match_frozen_audit": True}
    with (data/"train_index.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    labels = [np.load(data/f"train_pl{band}.npy", mmap_mode="r", allow_pickle=False)
              for band in ("415", "58")]
    masks = [np.load(data/f"train_mask{band}.npy", mmap_mode="r", allow_pickle=False)
             for band in ("415", "58")]

    physical, cache_sources = {}, {}
    wanted = set(ids)
    for folder in ("c73_compact", "c73_dev_extra_compact"):
        for directory in sorted((ROOT/"features"/folder).glob("part*")):
            if not (directory/"complete.json").exists():
                continue
            index = read(directory/"indices.json")
            matched = [(position, int(i)) for position, i in enumerate(index) if int(i) in wanted]
            if not matched:
                continue
            values = np.load(directory/"features.npy", mmap_mode="r", allow_pickle=False)
            assert values.dtype == np.float16 and values.shape[1:] == (45, 256, 256)
            for position, i in matched:
                assert i not in physical
                physical[i] = (values, position)
            cache_sources[str(directory.relative_to(ROOT))] = {
                "features": record(directory/"features.npy", hashed=False),
                "indices": record(directory/"indices.json"),
                "complete": record(directory/"complete.json"),
                "monitor_rows_used": len(matched),
                "feature_file_hash_recomputed_now": False}
    assert set(physical) == wanted
    assert summary["floor_definition"]["primary_channels"]["total"] == [41, 42]
    assert summary["floor_definition"]["floor_code"] == -3.0

    names = ("LoS_highest_m_lt_0", "NLoS_highest_m_ge_0",
             "GT_lt_90dB", "GT_ge_90_lt_110dB", "GT_ge_110_le_130dB", "GT_gt_130dB",
             "c73_total_floor_coded", "c73_total_above_floor_coded",
             "selected12_abs_error_ge_5dB", "selected12_abs_error_lt_5dB")
    groups = {name: Stats() for name in names}
    all_stats = Stats()
    terrains = {}
    terrain_maps = {}
    gt_stream_hash = [hashlib.sha256(), hashlib.sha256()]
    mask_stream_hash = [hashlib.sha256(), hashlib.sha256()]
    for position, i in enumerate(ids):
        label = np.stack([array[i] for array in labels]).astype(np.float64)
        valid = np.stack([np.unpackbits(array[i], bitorder="big").reshape(256, 256)
                          for array in masks]).astype(bool)
        pred = np.asarray(prediction[position], np.float64)
        assert np.isfinite(label[valid]).all() and np.isfinite(pred[valid]).all()
        error = pred-label
        squared = error*error
        highest = np.asarray(profiles[position, 0])
        assert np.isfinite(highest).all()
        compact, row = physical[i]
        floor = np.asarray(compact[row, [41, 42]], np.float32) <= -3.0
        conditions = (highest < 0, highest >= 0, label < 90,
                      (label >= 90) & (label < 110), (label >= 110) & (label <= 130),
                      label > 130, floor, ~floor, np.abs(error) >= 5, np.abs(error) < 5)
        all_stats.add(error, squared, valid)
        for name, condition in zip(names, conditions):
            groups[name].add(error, squared, valid & condition)
        terrain = rows[i]["terrain_id"]
        terrains.setdefault(terrain, Stats()).add(error, squared, valid)
        terrain_maps.setdefault(terrain, []).append(i)
        for band in range(2):
            gt_stream_hash[band].update(np.asarray(labels[band][i]).tobytes(order="C"))
            mask_stream_hash[band].update(np.asarray(masks[band][i]).tobytes(order="C"))
        if (position+1) % 128 == 0:
            print(json.dumps({"CPU_scored_maps": position+1, "total_maps": 628}), flush=True)
    assert len(terrains) == 128
    np.testing.assert_array_equal(all_stats.count, EXPECTED_COUNT)
    rmse = np.sqrt(all_stats.sse/all_stats.count)
    np.testing.assert_allclose(rmse, EXPECTED_RMSE, atol=1e-12, rtol=0)
    np.testing.assert_allclose(all_stats.sse, summary["all"]["SSE_dB2"], atol=1e-6, rtol=1e-12)
    partitions = {
        "LoS_NLoS": names[:2], "GT_pathloss": names[2:6],
        "c73_total_floor": names[6:8], "selected12_tail": names[8:10]}
    checks = {}
    for name, members in partitions.items():
        count = sum((groups[key].count for key in members))
        sse = sum((groups[key].sse for key in members))
        err = sum((groups[key].error_sum for key in members))
        np.testing.assert_array_equal(count, all_stats.count)
        np.testing.assert_allclose(sse, all_stats.sse, atol=1e-6, rtol=1e-12)
        np.testing.assert_allclose(err, all_stats.error_sum, atol=1e-6, rtol=1e-12)
        checks[name] = {"members": members, "count_sum": count.tolist(),
                        "SSE_sum_dB2": sse.tolist(),
                        "SSE_minus_all_dB2": (sse-all_stats.sse).tolist(),
                        "error_sum_minus_all_dB": (err-all_stats.error_sum).tolist()}
    old_group_map = {"LoS_highest_m_lt_0": "profile_LoS", "NLoS_highest_m_ge_0": "profile_NLoS",
                     "c73_total_floor_coded": "floor_total_linear_mean",
                     "selected12_abs_error_ge_5dB": "selected12_error_abs_ge_5dB"}
    for current, old in old_group_map.items():
        np.testing.assert_array_equal(groups[current].count, summary["groups"][old]["valid_pixels"])
        np.testing.assert_allclose(groups[current].sse, summary["groups"][old]["SSE_dB2"],
                                   atol=1e-6, rtol=1e-12)
    top5 = []
    for old in summary["worst_10_terrains_joint_MSE"][:5]:
        name = old["terrain"]
        value = terrains[name].value(all_stats)
        np.testing.assert_array_equal(value["valid_pixels"], old["valid_pixels"])
        np.testing.assert_allclose(value["SSE_dB2"], old["SSE_dB2"], atol=1e-6, rtol=1e-12)
        np.testing.assert_allclose(value["RMSE_dB"], old["RMSE_dB"], atol=1e-12, rtol=0)
        top5.append({"terrain": name, "monitor_map_count": len(terrain_maps[name]),
                     "monitor_map_ids": terrain_maps[name],
                     "joint_mean_MSE_dB2": old["joint_mean_MSE_dB2"], **value})
    assert all(sha(path) == digest for path, digest in frozen.items())
    assert "torch" not in sys.modules
    result = {
        "status": "complete_FP64_CPU_saved_prediction_supplement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": summary["scope"], "band_order": BANDS,
        "training_run": False, "network_inference_run": False, "geometry_run": False,
        "model_weights_read_or_changed": False, "SwanLab_run_created": False,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "torch_imported": False,
        "definitions": {
            "MeanBias": "sum(prediction_dB-GT_dB)/group_valid_count; positive=predicted pathloss too large",
            "RMSE": "sqrt(group_SSE/group_valid_count), with FP64 per-pixel errors and sums",
            "pixel_fraction_denominator": "same band's all valid monitor pixels, not all grid cells",
            "SSE_share_denominator": "same band's all valid monitor SSE",
            "LoS": "C149 cached profiles[:,0]=highest_m <0; NLoS >=0; not inverted c73 asinh proxy",
            "GT_buckets": {"GT_lt_90dB": "GT<90", "GT_ge_90_lt_110dB": "90<=GT<110",
                           "GT_ge_110_le_130dB": "110<=GT<=130", "GT_gt_130dB": "GT>130"},
            "GT_units": "Original TRAIN pathloss dB, separately bucketed in each frequency",
            "total_floor": summary["floor_definition"],
            "tail": "abs(saved C145 selected12 prediction - original GT)>=5dB; not C82-defined tail",
            "Top5": "Reuse frozen summary.worst_10_terrains_joint_MSE[:5]: descending arithmetic mean of two per-band terrain MSEs; NOT mean of RMSEs",
            "mask": "Original per-band big-endian packed validity; mask only scoring"},
        "all": all_stats.value(all_stats),
        "groups": {name: value.value(all_stats) for name, value in groups.items()},
        "worst_5_terrains_frozen_joint_MSE": top5,
        "validation": {"expected_count": EXPECTED_COUNT.tolist(),
                       "expected_RMSE_dB": EXPECTED_RMSE.tolist(),
                       "RMSE_minus_expected_dB": (rmse-EXPECTED_RMSE).tolist(),
                       "partitions": checks, "old_C149_group_counts_SSE_reconciled": True,
                       "frozen_summary_protocol_sources_unchanged": True},
        "source_identity": {
            "script": record(Path(__file__)), "C149_summary": record(summary_path),
            "C149_protocol": record(protocol_path), "C149_profiles": record(profiles_path, hashed=False),
            "C149_profiles_verified_sha256": summary["profile_cache_sha256"],
            "C145_verified": record(RUN/"selected_verified.json"),
            "prediction_export_validation": identity, "prediction": record(pred_path, hashed=False),
            "raw_audit": record(raw_audit_path), "raw_sources": raw_sources,
            "monitor_ordered_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "monitor_GT_row_stream_sha256": [value.hexdigest() for value in gt_stream_hash],
            "monitor_packed_mask_row_stream_sha256": [value.hexdigest() for value in mask_stream_hash],
            "row_stream_order": "C149 protocol ordered_panel_ids; each original stored row C-order bytes",
            "c73_sources": cache_sources},
        "elapsed_seconds": time.monotonic()-started}
    destination = OUT/"supplement_bias_gt_bins.json"
    if destination.exists():
        raise FileExistsError("Supplement already exists; inspect/reuse instead of overwriting")
    temporary = destination.with_suffix(".partial.json")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(destination)
    print(json.dumps({"status": result["status"], "output": str(destination),
                      "all": result["all"], "groups": result["groups"],
                      "top5": top5, "elapsed_seconds": result["elapsed_seconds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

