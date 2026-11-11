"""Export frozen C38 model predictions to the released Kaggle CSV template.

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
    parser.add_argument("--output", default="submissions/C38_Kaggle_20261102")
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay inside CC")
    out.mkdir(parents=True, exist_ok=False)
    template = DATA / "sample_submission.csv"
    template_sha = sha256(template)
    proof = json.loads((ROOT / "audit/c38_final_delivery.json").read_text())
    expected_weight_sha = proof["weights_sha256"]
    for weight in [ROOT / "runs/c38_full_refit/best.pt",
                   ROOT / "releases/C38_20261102/model/best.pt"]:
        assert sha256(weight) == expected_weight_sha, f"Changed checkpoint: {weight}"

    with (DATA / "val_index.csv").open(newline="") as f:
        val_rows = list(csv.DictReader(f))
    val_ids = {r["sample_id"] for r in val_rows}
    with (DATA / "subset_index.csv").open(newline="") as f:
        subset_ids = {r["sample_id"] for r in csv.DictReader(f)}

    per_map_root = ROOT / "audit/c38_full_refit_official/predictions"
    arrays = {}
    map_hashes = {}
    seen = set()
    band_counts = collections.Counter()
    map_counts = collections.Counter()
    partial = out / "submission_C38.csv.partial"
    final = out / "submission_C38.csv"

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
    for d in sorted((ROOT / "runs/c38_full_refit/predictions").glob("val_full_c34_fresh_shard*")):
        meta = json.loads((d / "inference.json").read_text())
        assert meta["no_target_or_validity_mask_read"] is True
        assert meta["run"] == "c38_full_refit" and meta["split"] == "val"
        indices = json.loads((d / "indices.json").read_text())
        assert len(indices["index_rows"]) == meta["maps_per_band"]
        for band in ("415", "58"):
            a = np.load(d / f"pl{band}.npy", mmap_mode="r", allow_pickle=False)
            assert a.shape == (meta["maps_per_band"], 256, 256)
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
        "model": "C38 full TRAIN refit, frozen best.pt at step 35540",
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
        "weights_sha256": expected_weight_sha,
        "export_script_sha256": sha256(__file__),
        "source_shards": source_shards,
        "source_per_map_sha256": map_hashes,
        "official_fixed_validation": proof["official"],
        "official_format_reference": "https://github.com/anderspearson206/lunar-radiomap-challenge#submitting",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    readme = """# C38 Kaggle 上传版本

上传文件：submission_C38.csv

在比赛页面点击 Submit Prediction，选择 submission_C38.csv。
提交说明可填写：C38 frozen full-TRAIN model; heightmap and TX inputs only.
文件不需要改名。上传 CSV，不要上传 manifest.json 或本说明。

本文件含 ID、PL 两列，共 800,000 行预测（另有一行表头）。
包含 100 个样本、两个频段；每个样本每个频段 4,000 个指定像素。
PL 是模型预测的路径损耗，单位 dB。没有用真实标签替换预测。

版本：C38。全部预测来自已冻结并核验的模型。
完整官方固定验证集，2,330 张图/频段：
- 415 MHz 有效像素 RMSE：2.6727180785 dB。
- 5.8 GHz 有效像素 RMSE：3.0805375616 dB。
- 双频 50/50 平均：2.8766278200 dB。

上述数值不是本次网站提交成绩。Kaggle 只对模板中的部分像素评分，
且公开榜单仅显示其中的公开划分，分数可能不同。
官方说明该 Kaggle CSV 榜单用于格式检查，不参与最终比赛排名。
这个 CSV 不是十二月隐藏测试集的最终模型提交包。

已检查：行数、全部 ID 及其顺序、重复 ID、有限数值，
并将全部 800,000 个数值与原始推理分片独立核对，float32 完全一致。
manifest.json 记录文件 SHA256、模板与模型来源。
本次复用已有模型预测，没有重新训练，没有使用 GPU，也没有进行网站提交。

官方格式说明：
https://github.com/anderspearson206/lunar-radiomap-challenge#submitting

服务器复现（从 CC 执行；输出目录必须尚不存在）：
.venv/bin/python scripts/export_c38_kaggle.py --output submissions/C38_Kaggle_reexport
"""
    (out / "README_上传说明.md").write_text(readme)
    print(json.dumps({k: manifest[k] for k in [
        "submission_file", "row_count", "bytes", "sha256",
        "PL_min_max_by_band", "max_float64_text_rounding_error_db",
        "all_values_match_original_inference_float32", "website_submission"
    ]}, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
