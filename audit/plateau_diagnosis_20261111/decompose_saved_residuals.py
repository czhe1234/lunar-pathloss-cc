"""Decompose SSE changes using frozen C101 monitor predictions. No fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np


CC = Path(__file__).resolve().parents[2]
MONITOR = CC / "audit/c101_residual/monitor"
FIT = CC / "audit/c101_residual/fit"
TAIL = CC / "audit/c125_tail"
SPECTRUM = CC / "runs/c124_spectrum"
GATE = CC / "audit/c118_location/location_monitor_probability.npy"
USED = set()


def load(path):
    USED.add(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def read(path):
    USED.add(path)
    return json.loads(path.read_text())


def score(residual, correction, selected):
    residual = residual[selected]
    correction = correction[selected]
    baseline_sse = float(np.dot(residual, residual))
    alignment = float(2 * np.dot(residual, correction))
    penalty = float(np.dot(correction, correction))
    error = residual - correction
    sse = float(np.dot(error, error))
    assert np.isclose(sse, baseline_sse - alignment + penalty, atol=1e-8)
    return {
        "n": len(residual),
        "baseline_sse": baseline_sse,
        "alignment_2_r_delta": alignment,
        "correction_cost_delta_sq": penalty,
        "after_sse": sse,
        "sse_removed": baseline_sse - sse,
    }


def confusion(actual, predicted):
    tp = int(np.count_nonzero(actual & predicted))
    pp = int(np.count_nonzero(predicted))
    positives = int(np.count_nonzero(actual))
    return {
        "true_positive": tp,
        "false_positive": pp - tp,
        "false_negative": positives - tp,
        "precision": tp / pp,
        "recall": tp / positives,
    }


def rmse(residual, correction):
    return float(np.sqrt(np.mean(np.square(residual - correction))))


def main():
    target = load(MONITOR / "target.npy")
    base = load(MONITOR / "base.npy")
    maps = load(MONITOR / "mapid.npy")
    prob = load(GATE)
    fit_target = load(FIT / "target.npy")
    fit_base = load(FIT / "base.npy")
    fit_weight = load(FIT / "weight.npy")
    gate_reference = read(TAIL / "frozen_gate_probe.json")["bands"]
    spectrum_reference = read(TAIL / "diagnosis.json")["bands"]
    assert target.shape == (628 * 2048, 4)
    assert base.shape == (len(target), 2) and prob.shape == (len(target), 2, 3)
    assert len(np.unique(maps)) == 628
    assert np.allclose(prob.sum(axis=-1), 1, rtol=0, atol=1e-5)

    results = []
    for band, frequency in enumerate((415e6, 5.775e9)):
        valid = target[:, band + 2] > 0
        rows = np.flatnonzero(valid)
        assert np.array_equal(rows, load(SPECTRUM / f"monitor_rows{band}.npy"))
        residual = target[valid, band].astype(float) - base[valid, band]
        tail = np.abs(residual) >= 5
        gate = (prob[valid, band, 0] + prob[valid, band, 2]).astype(float)
        conditional = load(TAIL / f"old_conditional_all{band}.npy")[valid]
        fit_residual = fit_target[:, band].astype(float) - fit_base[:, band]
        fit_non_tail = (fit_target[:, band + 2] > 0) & (np.abs(fit_residual) < 5)
        w = fit_weight[fit_non_tail]
        non_tail_mean = float(np.sum(fit_residual[fit_non_tail] * w) / np.sum(w))
        actual = gate * conditional + (1 - gate) * non_tail_mean
        saved_actual = load(TAIL / f"frozen_gate_delta{band}.npy")[valid]
        assert np.array_equal(actual.astype(np.float32), saved_actual)
        truth_gate = np.where(tail, conditional.astype(float), non_tail_mean)
        spectrum = load(SPECTRUM / f"monitor_delta{band}.npy").astype(float)
        assert spectrum.shape == residual.shape
        spectrum_correct_sign = (spectrum >= 0) == (residual > 0)

        verified_scores = {
            "C82": rmse(residual, 0),
            "C125_predicted_gate": rmse(residual, actual),
            "C125_truth_gate": rmse(residual, truth_gate),
            "C124": rmse(residual, spectrum),
        }
        ref = gate_reference[band]
        assert np.allclose(
            [verified_scores[k] for k in ("C82", "C125_predicted_gate", "C125_truth_gate", "C124")],
            [ref["C82_RMSE"], ref["actual_RMSE"], ref["true_membership_RMSE"],
             spectrum_reference[band]["C124_RMSE"]],
            atol=1e-11, rtol=0,
        )

        top = np.zeros_like(tail)
        n_top = int(np.ceil(tail.mean() * len(tail)))
        top[np.argsort(-gate)[:n_top]] = True
        thresholds = {
            "probability_ge_half": confusion(tail, gate >= .5),
            "top_count_equal_true_prevalence": confusion(tail, top),
        }
        assert np.isclose(thresholds["probability_ge_half"]["recall"], ref["tail_recall_at_gate_half"])
        assert np.isclose(thresholds["top_count_equal_true_prevalence"]["precision"], ref["precision_at_true_tail_prevalence"])

        actual_tail = score(residual, actual, tail)
        actual_other = score(residual, actual, ~tail)
        truth_tail = score(residual, truth_gate, tail)
        truth_other = score(residual, truth_gate, ~tail)
        results.append({
            "frequency_hz": int(frequency),
            "rmse_dB": verified_scores,
            "C125": {
                "predicted_gate_tail": actual_tail,
                "predicted_gate_nontail": actual_other,
                "truth_gate_tail": truth_tail,
                "truth_gate_nontail": truth_other,
                "extra_tail_sse_due_to_predicted_vs_truth_gate": actual_tail["after_sse"] - truth_tail["after_sse"],
                "extra_nontail_sse_due_to_predicted_vs_truth_gate": actual_other["after_sse"] - truth_other["after_sse"],
                "fixed_gate_confusion": thresholds,
                "actual_soft_gate_sse_by_probability_ge_half": {
                    "true_positive": score(residual, actual, tail & (gate >= .5)),
                    "false_positive": score(residual, actual, ~tail & (gate >= .5)),
                    "false_negative": score(residual, actual, tail & (gate < .5)),
                    "true_negative": score(residual, actual, ~tail & (gate < .5)),
                },
            },
            "C124": {
                "tail_correct_direction": score(residual, spectrum, tail & spectrum_correct_sign),
                "tail_incorrect_direction": score(residual, spectrum, tail & ~spectrum_correct_sign),
                "nontail": score(residual, spectrum, ~tail),
            },
        })
    hashes = {}
    for path in sorted(USED):
        with path.open("rb") as f:
            hashes[str(path.relative_to(CC))] = hashlib.file_digest(f, "sha256").hexdigest()
    output = {
        "scope": "C101 sampled TRAIN monitor, 128 terrain/628 maps, 2048 points per map before frequency mask",
        "tail": "absolute C82 error at least 5 dB",
        "method": "SSE removed = 2*sum(r*delta)-sum(delta^2), where r=label-C82; true tail membership is label-assisted",
        "provenance": "C125 saved gate correction reproduced bitwise after Float32 round-trip; C124 saved row IDs equal the complete valid row indices; all reference RMSEs reproduced",
        "bands": results,
        "sha256": hashes,
    }
    destination = Path(__file__).with_name("saved_residual_sse_decomposition.json")
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for band in results:
        print(band["frequency_hz"], band["rmse_dB"])
        for key in ("predicted_gate_tail", "predicted_gate_nontail", "truth_gate_tail", "truth_gate_nontail"):
            print(key, round(band["C125"][key]["sse_removed"], 2))
        for key in ("tail_correct_direction", "tail_incorrect_direction", "nontail"):
            print("C124", key, round(band["C124"][key]["sse_removed"], 2))
        print(band["C125"]["fixed_gate_confusion"])


if __name__ == "__main__":
    main()
