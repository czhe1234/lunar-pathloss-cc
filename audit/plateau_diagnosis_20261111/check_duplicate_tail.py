"""Compare C82 errors and label differences for the frozen C112 duplicate inputs."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
C112 = ROOT / "audit/c112_TXresponse"
DATA = ROOT.parent / "data"
OUTPUT = Path(__file__).with_name("duplicate_tail_check.json")


def load_predictions():
    predictions = {}
    for shard_id in (0, 1):
        shard = C112 / f"shard{shard_id}"
        ids = json.loads((shard / "ids.json").read_text())
        array = np.load(shard / "prediction.npy", mmap_mode="r")
        predictions.update(zip(ids, array, strict=True))
    return predictions


def group_sums(selected, ei, ej, gap):
    return {
        "paired_pixels": int(selected.sum()),
        "label_diff_sse_dB2": float(np.square(gap[selected]).sum()),
        "mean_c82_sse_dB2": float(((np.square(ei[selected]) + np.square(ej[selected])) / 2).sum()),
        "label_gap_at_least_1dB": int(np.count_nonzero(selected & (gap >= 1))),
        "label_gap_at_least_3dB": int(np.count_nonzero(selected & (gap >= 3))),
        "label_gap_at_least_5dB": int(np.count_nonzero(selected & (gap >= 5))),
    }


def group_summary(sums):
    count = sums["paired_pixels"]
    label_sse = sums["label_diff_sse_dB2"]
    model_sse = sums["mean_c82_sse_dB2"]
    return {
        **sums,
        "interlabel_rmse_dB": float(np.sqrt(label_sse / count)),
        "pair_average_c82_rmse_dB": float(np.sqrt(model_sse / count)),
        "best_shared_prediction_rmse_dB": float(np.sqrt(label_sse / (4 * count))),
        "empirical_duplicate_conflict_share_of_c82_sse": label_sse / (4 * model_sse),
    }


def evaluate_band(band, band_index, records, archived, rows, predictions):
    labels = np.load(DATA / f"train_pl{band}.npy", mmap_mode="r")
    masks = np.load(DATA / f"train_mask{band}.npy", mmap_mode="r")
    rx_row, rx_col = np.indices((256, 256))
    metrics_by_pair = []
    for record in records:
        i, j = record["ids"]
        keys = ("hm_row", "tx_row", "tx_col")
        assert all(rows[i][k] == rows[j][k] for k in keys)
        tx_row = int(rows[i]["tx_row"])
        tx_col = int(rows[i]["tx_col"])
        common = np.square(rx_row - tx_row) + np.square(rx_col - tx_col) >= 100
        for k in (i, j):
            common &= np.unpackbits(masks[k]).reshape(256, 256).astype(bool)
        assert np.count_nonzero(common) == record["bands"][band_index]["count"]
        yi, yj = (labels[k][common].astype(np.float64) for k in (i, j))
        pi, pj = (predictions[k][band_index][common].astype(np.float64) for k in (i, j))
        ei, ej = yi - pi, yj - pj
        gap = np.abs(yi - yj)
        tail_i, tail_j = np.abs(ei) >= 5, np.abs(ej) >= 5
        ordinary = (np.abs(ei) <= 1) & (np.abs(ej) <= 1)
        metrics_by_pair.append({
            "sample_ids": [rows[i]["sample_id"], rows[j]["sample_id"]],
            "tail_both": int(np.count_nonzero(tail_i & tail_j)),
            "tail_both_same_error_sign": int(np.count_nonzero(tail_i & tail_j & (ei * ej > 0))),
            "all": group_sums(np.ones_like(tail_i), ei, ej, gap),
            "tail_either": group_sums(tail_i | tail_j, ei, ej, gap),
            "ordinary_both": group_sums(ordinary, ei, ej, gap),
        })
    groups = {}
    for name in ("all", "tail_either", "ordinary_both"):
        sums = {key: sum(record[name][key] for record in metrics_by_pair)
                for key in metrics_by_pair[0][name]}
        groups[name] = group_summary(sums)
    assert groups["all"]["paired_pixels"] == archived["count"]
    assert np.isclose(groups["all"]["pair_average_c82_rmse_dB"], archived["base_permap_RMSE_on_commonpairs"])
    assert np.isclose(groups["all"]["interlabel_rmse_dB"], archived["label_delta_RMSE"])
    tail_both = sum(record["tail_both"] for record in metrics_by_pair)
    same_sign = sum(record["tail_both_same_error_sign"] for record in metrics_by_pair)
    return {
        "tail_both": tail_both,
        "tail_both_same_error_sign": same_sign,
        "tail_both_over_either": tail_both / groups["tail_either"]["paired_pixels"],
        "same_error_sign_over_tail_both": same_sign / tail_both,
        **groups,
        "by_pair": metrics_by_pair,
    }


def main():
    summary = json.loads((C112 / "summary.json").read_text())
    records = [record for record in summary["perpair"] if record["category"] == "exact0"]
    assert len(records) == len({r["terrain"] for r in records}) == 8
    with (DATA / "train_index.csv").open() as index:
        rows = list(csv.DictReader(index))
    predictions = load_predictions()
    archived = summary["summary"]["exact0"]["bands"]
    result = {
        "scope": "8 C112 official TRAIN pairs with the same released H and TX pixel; common-valid RX pixels >=10m from TX",
        "tail": "absolute C82 error >=5 dB on either record",
        "ordinary": "absolute C82 error <=1 dB on both records",
        "precision": "Released path loss is Float16. Reuse frozen development-C82 predictions and compute statistics in Float64.",
        "interpretation": "For two labels at one fixed input, their mean minimizes common-prediction MSE, which equals squared label gap / 4. This is the empirical conflict from those two records.",
        "limit": "Repeat scene and RNG independence is unverified. The check does not estimate official seed noise or a global irreducible error, and does not identify the origin of shared errors.",
        "sources": [
            "audit/c112_TXresponse/summary.json",
            "audit/c112_TXresponse/shard{0,1}/{ids.json,prediction.npy}",
            "../data/train_index.csv",
            "../data/train_{pl,mask}{415,58}.npy",
        ],
        "bands": {
            str(freq): evaluate_band(band, idx, records, archived[idx], rows, predictions)
            for idx, (band, freq) in enumerate((("415", 415000000), ("58", 5775000000)))
        },
    }
    tmp = OUTPUT.with_suffix(".partial")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(OUTPUT)
    for band, metrics in result["bands"].items():
        print(band, json.dumps({k: v for k, v in metrics.items() if k != "by_pair"}))


if __name__ == "__main__":
    main()
