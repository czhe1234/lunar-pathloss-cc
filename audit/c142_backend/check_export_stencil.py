"""Apply one archived C94 export stencil to frozen C141 FIT direct powers."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
C141 = ROOT / "audit/c141_direct_receiver"
STENCIL = ROOT / "audit/c94_measurement/templates_numpy126/sionna_centers_default.npz"
FREQUENCIES_HZ = np.array([415e6, 5.775e9])
WAVELENGTH_LOSS_DB = 20 * np.log10(4 * np.pi * FREQUENCIES_HZ / 299792458.0)
ACTIVE_WEIGHT = 1e-12
VIOLATION_DB = 1.0


def source(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def positive_donors(ids, weights):
    return {int(face) for face in ids[weights > ACTIVE_WEIGHT]}


def summarize(records):
    margins = [r["label_minus_weakest_direct_db"] for r in records if r["label_minus_weakest_direct_db"] is not None]
    return {
        "points": len(records),
        "same_positive_donor_set": sum(r["same_positive_donor_set"] for r in records),
        "uncached_alternate_donor": sum(bool(r["uncached_alternate_donors"]) for r in records),
        "zero_power_uninformative": sum(r["zero_power_uninformative"] is True for r in records),
        "finite_all_alternate_donors": len(margins),
        "finite_above_1_db_in_either_band": sum(any(x > VIOLATION_DB for x in p) for p in margins),
        "finite_above_1_db_in_both_bands": sum(all(x > VIOLATION_DB for x in p) for p in margins),
    }


def main():
    counterexamples = C141 / "remaining_finite_counterexamples.csv"
    with counterexamples.open() as handle:
        cases = {(int(row["map"]), 256 * int(row["row"]) + int(row["col"])) for row in csv.DictReader(handle)}
    maps = {mid for mid, _ in cases}
    panel = C141 / "fit_panel.json"
    inputs = [source(counterexamples), source(panel), source(STENCIL)]
    groups = {"counterexamples": [], "ordinary_controls_on_same_maps": []}
    with np.load(STENCIL) as stencil:
        alternate_ids, alternate_weights = stencil["ids"], stencil["weights"]

    for record in json.loads(panel.read_text())["records"]:
        mid = record["map"]
        if mid not in maps:
            continue
        path = C141 / "fit_replay" / f"{mid}.npz"
        inputs.append(source(path))
        with np.load(path) as replay:
            pixels = replay["pixels"]
            ids, weights = replay["ids"], replay["weights"]
            powers = replay["native_full_native"]
        assert np.array_equal(pixels, record["pixels"])
        assert np.all(np.isfinite(powers) & (powers >= 0))
        donor_power = {}
        for face, power in zip(ids.ravel(), powers.ravel()):
            assert donor_power.setdefault(int(face), float(power)) == power

        for position, pixel in enumerate(pixels):
            pixel = int(pixel)
            if (mid, pixel) in cases:
                group = "counterexamples"
                assert record["is_case"][position]
            elif not record["is_case"][position]:
                group = "ordinary_controls_on_same_maps"
            else:
                continue
            current = positive_donors(ids[position], weights[position])
            alternate = positive_donors(alternate_ids[pixel], alternate_weights[pixel])
            assert current and alternate
            unknown = alternate.difference(donor_power)
            margin = None
            zero = None
            if not unknown:
                weakest_power = min(donor_power[face] for face in alternate)
                zero = weakest_power == 0.0
                if not zero:
                    weakest_loss = WAVELENGTH_LOSS_DB - 10 * np.log10(weakest_power)
                    margin = (np.asarray(record["labels"][position]) - weakest_loss).tolist()
            row, col = divmod(pixel, 256)
            groups[group].append({
                "map": mid,
                "sample_id": record["sample_id"],
                "row": row,
                "col": col,
                "current_positive_donors": sorted(current),
                "alternate_positive_donors": sorted(alternate),
                "same_positive_donor_set": current == alternate,
                "uncached_alternate_donors": sorted(unknown),
                "zero_power_uninformative": zero,
                "label_minus_weakest_direct_db": margin,
            })

    assert len(groups["counterexamples"]) == len(cases) == 22
    assert len(maps) == 10 and len(groups["ordinary_controls_on_same_maps"]) == 80
    summary = {name: summarize(records) for name, records in groups.items()}
    result = {
        "scope": "Frozen C141 diagnostic FIT panel; 22 finite counterexamples and 80 ordinary controls on the same 10 terrain/TX maps",
        "frequencies_hz": FREQUENCIES_HZ.astype(int).tolist(),
        "active_weight_strictly_above": ACTIVE_WEIGHT,
        "violation_db_strictly_above": VIOLATION_DB,
        "alternate_stencil": "C94 archived default query grid with Sionna/DrJit centroids under NumPy 1.26.4; a candidate for the same physical receiver mesh",
        "sources": inputs,
        "summary": summary,
        "cases": groups["counterexamples"],
        "limits": [
            "Unknown donors were absent from the frozen C141 cache. No power was inferred for them.",
            "The bound covers any convex interpolation of the known positive-weight donors. It does not measure ordinary-point prediction RMSE.",
            "The stencil is an archived candidate. Exact competition export coordinates, donor selection, and ray-tracing backend remain unverified.",
            "This is cache-only CPU scoring. No ray tracing or validation data were used.",
        ],
    }
    out = Path(__file__).with_name("export_stencil_check.json")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
