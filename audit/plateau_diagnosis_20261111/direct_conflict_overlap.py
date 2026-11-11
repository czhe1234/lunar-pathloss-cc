"""Measure candidate direct-power conflict overlap with C82 error, using saved TRAIN-monitor caches."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MONITOR = ROOT / "audit/c101_residual/monitor"
DIRECT_CACHE = ROOT / "audit/c135_paths/cache"
C140 = ROOT / "audit/c140_consistency"
FREQUENCIES_HZ = np.array([415e6, 5.775e9])


def sum_pixels(values):
    return values.sum(axis=(0, 1))


def calculate():
    panel = json.loads((MONITOR / "done.json").read_text())
    ids = panel["ids"]
    points = panel["points_per_map"]
    assert len(ids) == 628 and points == 2048

    target = np.load(MONITOR / "target.npy", mmap_mode="r").reshape(len(ids), points, 4)
    pixels = np.load(MONITOR / "pixel.npy", mmap_mode="r").reshape(len(ids), points)
    c82 = np.load(MONITOR / "base.npy", mmap_mode="r").reshape(len(ids), points, 2).astype(float)
    y = target[..., :2].astype(float)
    mask = target[..., 2:]
    assert np.isin(mask, [0, 1]).all()
    valid = mask.astype(bool)
    gain_factor = (299792458. / (4 * np.pi * FREQUENCIES_HZ)) ** 2
    direct_limit = np.empty_like(y)
    for j, map_id in enumerate(ids):
        with np.load(DIRECT_CACHE / f"{map_id}.npz") as cache:
            assert np.array_equal(cache["pixels"], pixels[j]), map_id
            direct_gain = cache["power"][:, 0, :].astype(float) * gain_factor
        assert np.isfinite(direct_gain).all() and np.all(direct_gain >= 0)
        positive = direct_gain > 0
        direct_limit[j].fill(np.inf)
        direct_limit[j][positive] = -10 * np.log10(direct_gain[positive])

    c82_residual = y - c82
    c82_squared_error = np.square(c82_residual)
    direct_excess_db = np.maximum(y - direct_limit, 0.)
    direct_floor_squared_error = np.square(direct_excess_db)
    hard_projection = np.minimum(c82, direct_limit)
    hard_squared_error = np.square(y - hard_projection)
    hard_delta = hard_squared_error - c82_squared_error
    valid_count = sum_pixels(valid)
    c82_sse = sum_pixels(valid * c82_squared_error)
    hard_sse = sum_pixels(valid * hard_squared_error)
    direct_floor_sse = sum_pixels(valid * direct_floor_squared_error)
    c82_rmse = np.sqrt(c82_sse / valid_count)
    hard_rmse = np.sqrt(hard_sse / valid_count)
    direct_floor_rmse = np.sqrt(direct_floor_sse / valid_count)

    decision = json.loads((C140 / "decision.json").read_text())
    control = json.loads((C140 / "necessary_condition_control.json").read_text())
    np.testing.assert_array_equal(valid_count, control["C82"]["valid_count"])
    np.testing.assert_allclose(c82_rmse, control["C82"]["RMSE"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(hard_rmse, control["hard"]["RMSE"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(direct_floor_rmse, decision["direct_fixed_floor"], atol=1e-12, rtol=0)

    def summary(selected):
        weight = valid & selected
        count = sum_pixels(weight)
        error = sum_pixels(weight * c82_squared_error)
        floor = sum_pixels(weight * direct_floor_squared_error)
        return {
            "points": count.tolist(),
            "percent_valid_points": (100 * count / valid_count).tolist(),
            "C82_SSE_dB2": error.tolist(),
            "percent_C82_SSE": (100 * error / c82_sse).tolist(),
            "percent_direct_floor_SSE": (100 * floor / direct_floor_sse).tolist(),
            "hard_projection": {
                "improved_points": sum_pixels(weight & (hard_delta < 0)).tolist(),
                "worsened_points": sum_pixels(weight & (hard_delta > 0)).tolist(),
                "SSE_removed_dB2": sum_pixels(weight * np.maximum(-hard_delta, 0)).tolist(),
                "SSE_added_dB2": sum_pixels(weight * np.maximum(hard_delta, 0)).tolist(),
                "net_SSE_change_dB2": sum_pixels(weight * hard_delta).tolist(),
            },
        }

    def conflict_diagnostic(conflict):
        overlap = summary(conflict)
        overlap["oracle_RMSE_dB_if_C82_errors_on_these_points_perfectly_removed"] = np.sqrt(
            sum_pixels((valid & ~conflict) * c82_squared_error) / valid_count
        ).tolist()
        for name, tail in (
            ("all_tail", np.abs(c82_residual) >= 5),
            ("pathloss_underprediction_tail", c82_residual >= 5),
            ("pathloss_overprediction_tail", c82_residual <= -5),
        ):
            overlap[name + "_overlap_percent_SSE"] = (
                100 * sum_pixels((valid & conflict & tail) * c82_squared_error)
                / sum_pixels((valid & tail) * c82_squared_error)
            ).tolist()
        return overlap

    tail = np.abs(c82_residual) >= 5
    conflict = direct_excess_db > 1
    ordinary = np.abs(c82_residual) <= 1
    return {
        "scope": "Original 128-terrain TRAIN monitor, 628 maps, fixed 2048 sampled pixels per map.",
        "frequencies_hz": FREQUENCIES_HZ.tolist(),
        "valid_count": valid_count.tolist(),
        "definition": {
            "conflict": "Candidate direct gain exceeds the released total gain by more than 1 dB.",
            "direct_floor": "For fixed candidate direct power, the optimistic error of arbitrary nonnegative extra power, computed with labels.",
            "tail": "C82 absolute pathloss error >= 5 dB.",
            "ordinary": "C82 absolute pathloss error <= 1 dB.",
            "oracle": "Replace C82 with the released label only on the selected conflict pixels, preserve other C82 errors and use all valid points in the denominator. Diagnosis only.",
        },
        "source": {
            "C82_and_labels": str(MONITOR.relative_to(ROOT)),
            "candidate_direct_power": str(DIRECT_CACHE.relative_to(ROOT)),
            "reference_controls": str(C140.relative_to(ROOT)),
        },
        "reference_check": "Exact cache pixel alignment, valid counts, C82 RMSE, hard-projection RMSE and optimistic fixed-D floor verified against C140.",
        "C82_RMSE_dB": c82_rmse.tolist(),
        "candidate_direct_floor_RMSE_dB": direct_floor_rmse.tolist(),
        "hard_projection_RMSE_dB": hard_rmse.tolist(),
        "direct_conflict_over_1dB": conflict_diagnostic(conflict),
        "any_positive_direct_conflict_sensitivity": conflict_diagnostic(direct_excess_db > 0),
        "cross_tab": {
            "tail_with_conflict": summary(tail & conflict),
            "tail_without_conflict": summary(tail & ~conflict),
            "non_tail_with_conflict": summary(~tail & conflict),
            "non_tail_without_conflict": summary(~tail & ~conflict),
        },
        "ordinary_groups": {
            "all": summary(ordinary),
            "with_conflict": summary(ordinary & conflict),
            "without_conflict": summary(ordinary & ~conflict),
        },
        "hard_projection_by_C82_signed_tail": {
            "pathloss_underprediction_tail": summary(c82_residual >= 5),
            "pathloss_overprediction_tail": summary(c82_residual <= -5),
        },
    }


def main():
    result = calculate()
    destination = HERE / "direct_conflict_overlap.json"
    temporary = destination.with_suffix(".partial.json")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
