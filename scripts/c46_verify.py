"""Independently verify the selected C46 zero-correction fallback on CPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import hashlib, json
import numpy as np
import torch
R = Path(__file__).resolve().parents[1]
run = R / "runs/c46_residual_fit64"
audit = R / "audit/c46_residual_screen_20261102"
proto = json.loads((audit / "protocol.json").read_text())
result = json.loads((run / "result.json").read_text())
ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
assert ck["step"] == result["best_step"] == 0
head_keys = [f"readout.{b}.4.{s}" for b in range(2) for s in ["weight", "bias"]]
assert all(torch.count_nonzero(ck["model"][k]).item() == 0 for k in head_keys)
assert not result["official_VAL_loaded"]
sources = {}
for part in [0, 1]:
    root = R / "features/c46_base"
    meta = json.loads((root / f"fit_part{part}.json").read_text())
    assert meta["input_only"] and not meta["labels_masks_read"]
    a = np.load(root / f"fit_part{part}.npy", mmap_mode="r")
    for j, i in enumerate(meta["indices"]):
        sources[int(i)] = (a, j, False)
for root in sorted((R / "runs/c38_1024_continue/predictions").glob("train_dev_c34_fresh_shard*")):
    meta = json.loads((root / "inference.json").read_text())
    assert meta["no_target_or_validity_mask_read"]
    a = [np.load(root / f"pl{b}.npy", mmap_mode="r") for b in ["415", "58"]]
    for j, i in enumerate(json.loads((root / "indices.json").read_text())["index_rows"]):
        sources[int(i)] = (a, j, True)
Y = [np.load(R.parent / "data" / f"train_pl{b}.npy", mmap_mode="r") for b in ["415", "58"]]
M = [np.load(R.parent / "data" / f"train_mask{b}.npy", mmap_mode="r") for b in ["415", "58"]]
panels = {}
for panel, key in [("dev64_full", "DEV64_full"), ("fit_panel_full", "FIT_panel_full")]:
    meta = json.loads((run / f"{panel}.json").read_text())
    ids = meta["indices"]
    if panel == "dev64_full":
        assert ids == proto["dev64_ids"]
    pred = np.load(run / f"{panel}.npy", mmap_mode="r")
    assert pred.shape == (len(ids), 65536, 2)
    sse = np.zeros(2, dtype=np.float64)
    count = np.zeros(2, dtype=np.int64)
    max_difference = 0.
    for j, i in enumerate(ids):
        array, k, separate = sources[int(i)]
        base = np.stack([v[k] for v in array], axis=-1) if separate else np.moveaxis(array[k], 0, -1)
        base = base.reshape(65536, 2)
        assert np.isfinite(pred[j]).all()
        assert np.array_equal(pred[j], base), f"C46 != frozen C38 on row {i}"
        max_difference = max(max_difference, float(np.max(np.abs(pred[j] - base))))
        for b in range(2):
            keep = np.unpackbits(M[b][i], bitorder="big").reshape(256, 256).astype(bool)
            e = pred[j, :, b].reshape(256, 256)[keep].astype(np.float64) - Y[b][i][keep].astype(np.float64)
            sse[b] += e @ e
            count[b] += int(keep.sum())
    rmse = np.sqrt(sse / count)
    assert np.max(np.abs(rmse - np.array(result[key]["RMSE"]))) < 1e-9
    assert count.tolist() == result[key]["count"]
    panels[panel] = {"RMSE": rmse.tolist(), "combined": float(rmse.mean()),
                     "SSE": sse.tolist(), "count": count.tolist(), "maps": len(ids),
                     "all_pixels_equal_frozen_C38": True,
                     "maximum_absolute_prediction_difference_db": max_difference}
history = json.loads((run / "history.json").read_text())
initial, last = history[0], history[-1]
assert initial["step"] == 0 and last["step"] == 1000
out = {
    "selected_step": 0, "last_training_step": 1000,
    "head_keys": head_keys, "selected_output_heads_exactly_zero": True,
    "panels": panels, "independent_float64_masked_scoring": True,
    "initial_monitor": {"FIT": initial["FIT"]["RMSE"], "DEV": initial["DEV"]["RMSE"]},
    "last_monitor": {"FIT": last["FIT"]["RMSE"], "DEV": last["DEV"]["RMSE"]},
    "last_monitor_DEV_change_db": (np.array(last["DEV"]["RMSE"]) - initial["DEV"]["RMSE"]).tolist(),
    "last_monitor_FIT_change_db": (np.array(last["FIT"]["RMSE"]) - initial["FIT"]["RMSE"]).tolist(),
    "both_DEV_gains_at_least_0_10db": False,
    "decision": "Stop this residual-corrector screen; retain C38. No 548-terrain confirmation or official VAL run.",
    "FP32_field_replay": "Unnecessary for selected exact-zero output heads; saved outputs bitwise equal frozen C38 on all panel pixels.",
    "official_VAL_loaded": False, "website_submission": False,
    "best_checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
    "limits": "64 FIT terrains; one seed; 1000 steps. Does not exclude longer or differently designed nonlinear correctors."
}
(audit / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ["selected_step", "panels", "last_monitor_DEV_change_db", "last_monitor_FIT_change_db", "decision"]}, ensure_ascii=False, indent=2))
