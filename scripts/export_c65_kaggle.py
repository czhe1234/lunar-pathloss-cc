"""Export frozen C65 model predictions to the released Kaggle CSV template.

CPU only. Reads no target arrays or masks. Verifies every CSV value against
the original input-only inference shards, independently of per-map exports.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import collections
import csv
import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submissions/C65_Kaggle_20261102")
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay inside CC")
    out.mkdir(parents=True, exist_ok=False)
    template = DATA / "sample_submission.csv"
    template_sha = sha256(template)
    proof = json.loads((ROOT / "audit/c65_final_delivery.json").read_text())
    expected_weight_sha = proof["weights_manifest_sha256"]
    assert sha256(ROOT / "runs/c65_combo_full/manifest.json") == expected_weight_sha
    for role, info in proof["weights_roles"].items():
        for directory in [ROOT / "runs/c65_combo_full", ROOT / "releases/C65_20261102/model"]:
            assert sha256(directory / (role + ".pt")) == info["sha256"]

    with (DATA / "val_index.csv").open(newline="") as f:
        val_rows = list(csv.DictReader(f))
    val_ids = {r["sample_id"] for r in val_rows}
    with (DATA / "subset_index.csv").open(newline="") as f:
        subset_ids = {r["sample_id"] for r in csv.DictReader(f)}

    per_map_root = ROOT / "audit/c65_combo_full_official/predictions"
    arrays = {}
    map_hashes = {}
    seen = set()
    band_counts = collections.Counter()
    map_counts = collections.Counter()
    partial = out / "submission_C65.csv.partial"
    final = out / "submission_C65.csv"

    with template.open(newline="") as fi, partial.open("x", newline="") as fo:
        reader = csv.DictReader(fi)
        assert reader.fieldnames == ["ID", "PL"]
        writer = csv.writer(fo, lineterminator="\n")
        writer.writerow(["ID", "PL"])
        for row in reader:
            rid = row["ID"]
            assert rid not in seen, f"Duplicate ID: {rid}"
            seen.add(rid)
            head, pixel_text = rid.rsplit("_", 1)
            prefix, sid = head.split("_", 1)
            assert prefix in {"pl415", "pl58"} and sid in val_ids
            pixel = int(pixel_text)
            assert 0 <= pixel < 256 * 256
            key = (prefix, sid)
            if key not in arrays:
                p = per_map_root / f"{prefix}_{sid}.npy"
                array = np.load(p, mmap_mode="r", allow_pickle=False)
                assert array.shape == (256, 256) and array.dtype == np.float32
                assert np.isfinite(array).all()
                arrays[key] = array.reshape(-1)
                map_hashes[p.name] = sha256(p)
            value = float(arrays[key][pixel])
            writer.writerow([rid, format(value, ".9g")])
            band_counts[prefix] += 1
            map_counts[f"{prefix}_{sid}"] += 1
    assert {sid for _, sid in arrays} == subset_ids
    assert len(seen) == 800000 and band_counts == {"pl415": 400000, "pl58": 400000}
    assert len(arrays) == 200 and set(map_counts.values()) == {4000}
    print(json.dumps({"stage": "exported", "rows": len(seen),
                      "maps": len(arrays), "bytes": partial.stat().st_size}), flush=True)

    # Independent verification: obtain predictions from original inference
    # shards and index by explicit (row, column), rather than the per-map files.
    reference = {}
    source_shards = {}
    for d in sorted((ROOT / "runs/c65_combo_full/predictions").glob("val_full_c34_fresh_shard*")):
        meta = json.loads((d / "inference.json").read_text())
        assert meta["no_target_or_validity_mask_read"] is True
        assert meta["input_only"] and meta["complete"]
        indices = json.loads((d / "indices.json").read_text())
        assert len(indices["index_rows"]) == meta["maps"]
        for band in ("415", "58"):
            a = np.load(d / f"pl{band}.npy", mmap_mode="r", allow_pickle=False)
            assert a.shape == (meta["maps"], 256, 256)
            for j, i in enumerate(indices["index_rows"]):
                sid = val_rows[i]["sample_id"]
                assert sid == indices["sample_ids"][j]
                key = (band, sid)
                assert key not in reference
                reference[key] = (a, j)
        source_shards[d.name] = {
            "indices_sha256": sha256(d / "indices.json"),
            "inference_sha256": sha256(d / "inference.json"),
            "no_target_or_validity_mask_read": True,
        }
    assert len(reference) == 4660
    pattern = re.compile(r"pl(415|58)_([0-9]+_[0-9]+)_([0-9]+)")
    limits = {"415": [float("inf"), -float("inf")],
              "58": [float("inf"), -float("inf")]}
    max_text_error = 0.0
    verified = 0
    with template.open(newline="") as ft, partial.open(newline="") as fc:
        rt, rc = csv.DictReader(ft), csv.DictReader(fc)
        assert rt.fieldnames == rc.fieldnames == ["ID", "PL"]
        for expected_row, actual_row in itertools.zip_longest(rt, rc):
            assert expected_row is not None and actual_row is not None
            assert expected_row["ID"] == actual_row["ID"], "ID order changed"
            match = pattern.fullmatch(actual_row["ID"])
            assert match is not None
            band, sid, pixel_text = match.groups()
            array, j = reference[(band, sid)]
            row_i, col_i = divmod(int(pixel_text), 256)
            original = array[j, row_i, col_i]
            parsed = float(actual_row["PL"])
            assert np.isfinite(parsed)
            assert np.float32(parsed) == original, "Prediction value mismatch"
            max_text_error = max(max_text_error, abs(parsed - float(original)))
            limits[band][0] = min(limits[band][0], parsed)
            limits[band][1] = max(limits[band][1], parsed)
            verified += 1
    assert verified == len(seen)
    assert sha256(template) == template_sha
    partial.rename(final)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "C65 fixed-band fullTRAIN refit; low/high fixed17770steps; frozen fullTRAIN C38 teacher",
        "purpose": "Kaggle public CSV format-check leaderboard",
        "submission_file": final.name,
        "bytes": final.stat().st_size,
        "sha256": sha256(final),
        "columns": ["ID", "PL"],
        "row_count": verified,
        "band_row_counts": dict(band_counts),
        "sample_count": len(subset_ids),
        "band_map_count": len(arrays),
        "pixels_per_band_map": 4000,
        "template": str(template),
        "template_sha256": template_sha,
        "template_ID_sequence_exact": True,
        "duplicate_ID_count": 0,
        "all_values_finite": True,
        "unit": "dB pathloss",
        "PL_min_max_by_band": limits,
        "all_values_match_original_inference_float32": True,
        "max_float64_text_rounding_error_db": max_text_error,
        "training_performed_for_export": False,
        "GPU_used_for_export": False,
        "ground_truth_or_masks_read_for_export": False,
        "website_submission": False,
        "weights_manifest_sha256": expected_weight_sha,
        "export_script_sha256": sha256(__file__),
        "source_shards": source_shards,
        "source_per_map_sha256": map_hashes,
        "official_fixed_validation": proof["official"],
        "official_format_reference": "https://github.com/anderspearson206/lunar-radiomap-challenge#submitting",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    values = proof["official"]["RMSE"]
    readme = f"""# C65 上传说明

上传文件：submission_C65.csv。另有一行表头，共800000行预测，列为ID、PL。
本文件包含100个公开VAL子集样本、两个频段，每频段每图4000个模板像素。
所有PL均来自冻结模型预测，没有使用真实标签或掩码替换预测。

本地执行官方原版评分脚本，完整官方固定VAL每频段2330张图：
415MHz：{values[0]:.10f} dB；5.775GHz：{values[1]:.10f} dB。
综合平均：{proof["official"]["combined_rmse_db_masked"]:.10f} dB。
双频均≤1dB目标尚未达到，当前不是大幅提分版本。
这些数值不是新的网站提交成绩。

C65以全部官方TRAIN训练，包含原内部DEV；官方VAL未用于梯度、选步数或选择频段路由。
官方VAL在权重冻结后用于本地评分。CSV只从已保存的输入推理结果导出。
导出本身仅用CPU，没有重新训练，没有读取真实标签或有效掩码，没有网站提交。
全部ID、顺序及800000个数值已与原始推理分片独立核对，float32完全一致。

Kaggle公开CSV榜用于格式检查，不能代替隐藏测试集最终结果。
本CSV不是隐藏测试集的最终模型包。旧C38上传文件保留。
在网站Submit Prediction中选择本CSV；本次没有代为上传。
"""
    (out / "README_上传说明.md").write_text(readme)
    print(json.dumps({k: manifest[k] for k in [
        "submission_file", "row_count", "bytes", "sha256",
        "PL_min_max_by_band", "max_float64_text_rounding_error_db",
        "all_values_match_original_inference_float32", "website_submission"
    ]}, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
