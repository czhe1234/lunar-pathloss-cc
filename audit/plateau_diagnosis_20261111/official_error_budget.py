"""Read-only scoring of frozen C82 on the complete official fixed validation set."""
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data"
FROZEN = ROOT / "audit/c82_selected_official"
OUT = Path(__file__).with_suffix(".json")
BANDS = ("415", "58")
EDGES = np.array([1., 2., 3., 5., 10., 20.])


def band_summary(den, count, sse, signed_tail, terrain_sse, group):
    total = sse.sum()
    order = np.sort(terrain_sse)[::-1]
    cumulative = np.cumsum(order)
    threshold_records = []
    for edge_index, edge in enumerate(EDGES, start=1):
        count_ge = count[edge_index:].sum()
        sse_ge = sse[edge_index:].sum()
        threshold_records.append({
            "abs_error_ge_db": edge,
            "pixels": int(count_ge),
            "fraction_valid_pixels": count_ge / den,
            "fraction_C82_SSE": sse_ge / total,
            "oracle_RMSE_if_perfect_only_on_those_pixels_db": np.sqrt((total-sse_ge)/den),
        })
    # The oracle fixes all pixels of the terrains with the largest C82 SSE.
    territory = []
    for fraction in (.01, .05, .10, .20):
        n = math.ceil(fraction * len(order))
        territory.append({
            "terrain_count": n,
            "fraction_C82_SSE": cumulative[n-1] / total,
            "oracle_RMSE_after_fixing_these_terrains_db": np.sqrt((total-cumulative[n-1])/den),
        })
    min_terrains = int(np.searchsorted(cumulative, total-den)+1)
    return {
        "valid_pixels": int(den),
        "sse": total,
        "rmse_db": np.sqrt(total/den),
        "sse_reduction_required_for_1dB": 1-den/total,
        "thresholds": threshold_records,
        "count_abs_error_lt_1dB": int(count[0]),
        "abs_error_ge_5dB_C82_loss_too_low_count": int(signed_tail[0]),
        "abs_error_ge_5dB_C82_loss_too_high_count": int(signed_tail[1]),
        "top_terrain_oracles": territory,
        "fewest_perfectly_corrected_terrains_for_1dB": min_terrains,
        "terrain_groups": {
            name: {"valid_pixels": int(n), "rmse_db": np.sqrt(error/n),
                   "fraction_C82_SSE": error/total}
            for name, (n, error) in group.items()
        },
    }


def main():
    with (DATA / "val_index.csv").open() as file:
        rows = list(csv.DictReader(file))
    ref = json.loads((FROZEN / "verified_summary.json").read_text())
    pl = [np.load(DATA / f"val_pl{band}.npy", mmap_mode="r") for band in BANDS]
    mask = [np.load(DATA / f"val_mask{band}.npy", mmap_mode="r") for band in BANDS]
    assert len(rows) == 2330
    assert all(x.shape == (len(rows), 256, 256) for x in pl)
    assert all(x.shape == (len(rows), 8192) for x in mask)
    terrain_ids = sorted({row["terrain_id"] for row in rows})
    assert len(terrain_ids) == 680
    terrain_ix = {value: i for i, value in enumerate(terrain_ids)}
    terrain_sse = np.zeros((len(terrain_ids), 2))
    terrain_valid = np.zeros((len(terrain_ids), 2), dtype=np.int64)
    terrain_tail = np.zeros((len(terrain_ids), 2), dtype=np.int64)
    counts = np.zeros((2, len(EDGES)+1), dtype=np.int64)
    sse = np.zeros_like(counts, dtype=np.float64)
    signed_tail = np.zeros((2, 2), dtype=np.int64)
    groups = {name: np.zeros((2, 2)) for name in {row["group"] for row in rows}}
    common_valid = 0
    common_tail = np.zeros(2, dtype=np.int64)
    both_tail = 0
    both_tail_same_sign = 0
    for index, row in enumerate(rows):
        t = terrain_ix[row["terrain_id"]]
        valid = [np.unpackbits(arr[index]).astype(bool) for arr in mask]
        residual = []
        for b, band in enumerate(BANDS):
            p = np.load(FROZEN / "predictions" / f"pl{band}_{row['sample_id']}.npy")
            assert p.shape == (256, 256)
            r = pl[b][index].astype(np.float64).ravel()-p.ravel()
            r_valid = r[valid[b]]
            assert np.isfinite(r_valid).all()
            bucket = np.searchsorted(EDGES, np.abs(r_valid), side="right")
            square = r_valid*r_valid
            counts[b] += np.bincount(bucket, minlength=len(EDGES)+1)
            sse[b] += np.bincount(bucket, weights=square, minlength=len(EDGES)+1)
            terrain_valid[t, b] += len(r_valid)
            terrain_sse[t, b] += square.sum()
            terrain_tail[t, b] += int(np.count_nonzero(bucket >= 4))
            signed_tail[b] += [np.count_nonzero(r_valid >= 5), np.count_nonzero(r_valid <= -5)]
            groups[row["group"]][b] += [len(r_valid), square.sum()]
            residual.append(r)
        common = valid[0] & valid[1]
        tails = [common & (np.abs(r) >= 5) for r in residual]
        both = tails[0] & tails[1]
        common_valid += int(common.sum())
        common_tail += [x.sum() for x in tails]
        both_tail += int(both.sum())
        both_tail_same_sign += int(np.count_nonzero(both & ((residual[0] > 0) == (residual[1] > 0))))
    den = terrain_valid.sum(0)
    assert np.array_equal(den, counts.sum(1)) and np.array_equal(den, ref["valid_pixels"])
    assert np.allclose(terrain_sse.sum(0), sse.sum(1), rtol=0, atol=1e-5)
    assert np.allclose(np.sqrt(sse.sum(1)/den), ref["RMSE"], rtol=0, atol=1e-7)
    result = {
        "scope": "2330 full paired C82 frozen validation maps, 680 terrains, official per-band validity masks",
        "frequency_hz": [415_000_000, 5_775_000_000],
        "diagnostic_only": "Frozen predictions and labels used for decomposition only. No training, model choice or tuning.",
        "bands": [
            {
                **band_summary(den[b], counts[b], sse[b], signed_tail[b], terrain_sse[:, b],
                               {name: group[b] for name, group in groups.items()}),
                "terrains_with_valid_abs_error_ge_5dB": int(np.count_nonzero(terrain_tail[:, b])),
            }
            for b in range(2)
        ],
        "dual_band_tail_overlap": {
            "common_valid_pixels": common_valid,
            "tail_at_least_5dB_each_band_on_common_valid": common_tail.tolist(),
            "tail_in_both_bands": both_tail,
            "tail_in_both_as_fraction_of_each_band_tail": (both_tail/common_tail).tolist(),
            "same_sign_among_both_tail": both_tail_same_sign,
            "same_sign_fraction_among_both_tail": both_tail_same_sign/both_tail,
        },
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for band in result["bands"]:
        print(band["rmse_db"], band["thresholds"][3], band["top_terrain_oracles"], flush=True)
    print(result["dual_band_tail_overlap"], flush=True)


if __name__ == "__main__":
    main()
