"""Bounded C148 comparison and optional export; never rerun C145 training/evaluation.

Summary is CPU-only. Export is explicit, uses the protocol's full GPU UUID, and
skips inference for the existing verified C145 parent or a valid cached export.
"""
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "audit/c148_generalization"
RUN = ROOT / "runs/c148_parameter_anchor"
PARENT = ROOT / "runs/c145_capacity_native"
GROUPS = {"all": "all", "tail": "C82_abs_error_ge_5dB",
          "C82_good": "C82_abs_error_le_1dB"}
CHECKPOINTS = {"selected": "best_monitor.pt", "last": "last.pt"}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def context():
    plan_path = AUDIT / "protocol.json"
    plan = load(plan_path)
    if file_hash(ROOT / plan["base_protocol"]) != plan["base_protocol_sha256"]:
        raise ValueError("Original C145 protocol identity changed")
    for name, expected in plan["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    if file_hash(ROOT / plan["parent_checkpoint"]) != plan["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint identity changed")
    config = load(RUN / "config.json")
    result = load(RUN / "training_result.json")
    if result.get("status") != "complete" or not result.get("fixed_budget_complete"):
        raise ValueError("C148 fixed training budget is not complete")
    for value in (config, result):
        if value["protocol_sha256"] != file_hash(plan_path):
            raise ValueError("C148 run protocol identity differs")
    base = load(ROOT / plan["base_protocol"])
    if config["panels"] != base["panels"]:
        raise ValueError("C148 ordered panels differ from C145")
    if config.get("official_VAL_used") or result.get("official_VAL_used"):
        raise ValueError("This development comparison must not use official VAL")
    return plan, config, result


def group_metrics(group):
    sse = np.asarray(group["SSE"], np.float64)
    counts = np.asarray(group["count"], np.int64)
    baseline_sse = np.asarray(group["C82_SSE"], np.float64)
    rmse = np.asarray(group["RMSE"], np.float64)
    baseline = np.asarray(group["C82_RMSE"], np.float64)
    if any(value.shape != (2,) for value in (sse, counts, baseline_sse, rmse, baseline)):
        raise ValueError("Expected one same-checkpoint two-band metric")
    if np.any(counts <= 0) or not all(np.isfinite(value).all() for value in (sse, baseline_sse, rmse, baseline)):
        raise ValueError("Invalid full-panel group statistics")
    if not np.allclose(np.sqrt(sse / counts), rmse, atol=1e-10, rtol=0):
        raise ValueError("History RMSE differs from SSE/count")
    if not np.allclose(np.sqrt(baseline_sse / counts), baseline, atol=1e-10, rtol=0):
        raise ValueError("History C82 RMSE differs from SSE/count")
    return {"RMSE_dB": rmse.tolist(), "C82_RMSE_dB": baseline.tolist(),
            "SSE_dB2": sse.tolist(), "C82_SSE_dB2": baseline_sse.tolist(),
            "valid_pixels": counts.tolist(),
            "gain_C82_minus_candidate_dB": (baseline-rmse).tolist()}


def row_metrics(row):
    return {panel: {name: group_metrics(row["evaluation"][panel]["groups"][name])
                    for name in GROUPS} for panel in ("fit", "monitor")}


def compare_rows(candidate, reference):
    candidate_metrics = row_metrics(candidate)
    reference_metrics = row_metrics(reference)
    delta = {}
    for panel in candidate_metrics:
        delta[panel] = {}
        for name in GROUPS:
            current = candidate_metrics[panel][name]
            old = reference_metrics[panel][name]
            if current["valid_pixels"] != old["valid_pixels"]:
                raise ValueError(f"Comparison count differs: {panel}/{name}")
            if not np.allclose(current["C82_SSE_dB2"], old["C82_SSE_dB2"], atol=1e-5, rtol=1e-12):
                raise ValueError(f"Comparison baseline differs: {panel}/{name}")
            gain = np.asarray(old["RMSE_dB"])-np.asarray(current["RMSE_dB"])
            delta[panel][name] = {"gain_reference_minus_C148_dB": gain.tolist(),
                                 "C148_better_in_both_bands": bool(np.all(gain > 0))}
    return delta


def verified_evidence(tag, model_digest, expected_row, config):
    source = PARENT / "selected_verified.json" if model_digest == load(AUDIT / "protocol.json")["anchor_model_sha256"] else RUN / f"{tag}_verified.json"
    if not source.exists():
        return {"status": "pending", "required": str(source.relative_to(ROOT)),
                "command": f"CUDA_VISIBLE_DEVICES='' .venv/bin/python -m src.c145_verify --run runs/c148_parameter_anchor --tag {tag}"}
    verified = load(source)
    if verified.get("status") != "verified" or verified["export_identity"]["model_sha256"] != model_digest:
        raise ValueError(f"Independent verification belongs to another model: {source}")
    if source.parent == RUN and verified["config_sha256"] != file_hash(RUN / "config.json"):
        raise ValueError("Independent verification config identity changed")
    if verified.get("official_VAL_used"):
        raise ValueError("Unexpected official VAL in verification")
    for panel in ("fit", "monitor"):
        for history_group, verified_group in GROUPS.items():
            expected = group_metrics(expected_row["evaluation"][panel]["groups"][history_group])
            actual = verified["panels"][panel]["groups"][verified_group]
            if expected["valid_pixels"] != actual["valid_pixels"]:
                raise ValueError("Independent verifier valid count differs")
            if not np.allclose(expected["RMSE_dB"], actual["RMSE_dB"], rtol=0, atol=1e-9):
                raise ValueError("Independent verifier RMSE differs")
    return {"status": "verified", "source": str(source.relative_to(ROOT)),
            "source_sha256": file_hash(source),
            "reused_parent_without_new_inference": source.parent == PARENT}


def bootstrap_gain(parent_sse, candidate_sse, counts, *, seed=145073, replicates=2000):
    """Diagnostic paired terrain bootstrap, with one shared draw for both models/bands."""
    parent_sse = np.asarray(parent_sse, np.float64)
    candidate_sse = np.asarray(candidate_sse, np.float64)
    counts = np.asarray(counts, np.float64)
    if parent_sse.shape != candidate_sse.shape or parent_sse.shape != counts.shape:
        raise ValueError("Paired terrain bootstrap arrays differ")
    if parent_sse.ndim != 2 or parent_sse.shape[1] != 2 or len(parent_sse) == 0:
        raise ValueError("Expected terrain by two-band sufficient statistics")
    if not all(np.isfinite(x).all() for x in (parent_sse, candidate_sse, counts)):
        raise ValueError("Nonfinite terrain sufficient statistics")
    if np.any(counts < 0) or np.any(counts.sum(0) <= 0):
        raise ValueError("Invalid paired terrain counts")
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(parent_sse), size=(replicates, len(parent_sse)))
    denominator = counts[samples].sum(1)
    if np.any(denominator <= 0):
        raise ValueError("Bootstrap draw has no valid pixels")
    parent_rmse = np.sqrt(parent_sse[samples].sum(1) / denominator)
    candidate_rmse = np.sqrt(candidate_sse[samples].sum(1) / denominator)
    gain = parent_rmse-candidate_rmse
    return np.quantile(gain, [.025, .975], axis=0).T.tolist()


@lru_cache(maxsize=2)
def saved_monitor_statistics(path):
    """Read each immutable completed NPZ once; optional map fields may be absent."""
    optional = ("map_ids", "map_terrain_index", "map_group_sse",
                "map_group_baseline_sse", "map_group_count")
    with np.load(path, allow_pickle=False) as source:
        result = {name: source["monitor_terrain_" + name].copy()
                  for name in ("names", "sse", "C82_sse", "count")}
        result["missing_map_fields"] = ["monitor_" + name for name in optional
                                       if "monitor_" + name not in source]
        for name in optional:
            if "monitor_" + name in source:
                result[name] = source["monitor_" + name].copy()
        if "group_names" in source:
            result["group_names"] = source["group_names"].copy()
        else:
            result["missing_map_fields"].append("group_names")
    return result


def parent_gain_uncertainty(evidence, model_digest, expected_row):
    common = {"unit": "terrain, retaining all maps/pixels for each sampled terrain",
              "seed": 145073, "replicates": 2000,
              "definition": "RMSE(selected12)-RMSE(C148), positive favors C148; band rows, lower/upper columns",
              "selection_use": False,
              "method_limit": "Diagnostic conditional on repeatedly used development monitor and checkpoint selection; not a blind or selection-corrected confidence interval."}
    if evidence["status"] != "verified":
        return {**common, "status": "pending_independent_prediction_verification"}
    parent_path = PARENT / "selected_terrain_sse.npz"
    candidate_path = parent_path if model_digest == load(AUDIT / "protocol.json")["anchor_model_sha256"] else RUN / "selected_terrain_sse.npz"
    parent, candidate = (saved_monitor_statistics(path) for path in (parent_path, candidate_path))
    if not np.array_equal(parent["names"], candidate["names"]):
        raise ValueError("Parent/C148 ordered monitor terrain names differ")
    if not np.array_equal(parent["count"], candidate["count"]):
        raise ValueError("Parent/C148 monitor terrain counts differ")
    if not np.allclose(parent["C82_sse"], candidate["C82_sse"], atol=1e-5, rtol=1e-12):
        raise ValueError("Parent/C148 monitor C82 terrain SSE differs")
    group = group_metrics(expected_row["evaluation"]["monitor"]["groups"]["all"])
    if not np.array_equal(candidate["count"].sum(0), group["valid_pixels"]):
        raise ValueError("Candidate terrain counts differ from full-panel counts")
    if not np.allclose(np.sqrt(candidate["sse"].sum(0)/candidate["count"].sum(0)),
                       group["RMSE_dB"], atol=1e-9, rtol=0):
        raise ValueError("Candidate terrain SSE differs from independent full-panel result")
    verified_parent = load(PARENT / "selected_verified.json")["panels"]["monitor"]["groups"]["all"]
    if not np.array_equal(parent["count"].sum(0), verified_parent["valid_pixels"]):
        raise ValueError("Parent terrain counts differ from its verified panel")
    if not np.allclose(np.sqrt(parent["sse"].sum(0)/parent["count"].sum(0)),
                       verified_parent["RMSE_dB"], atol=1e-9, rtol=0):
        raise ValueError("Parent terrain SSE differs from its independent panel")
    same = candidate_path == parent_path
    interval = [[0., 0.], [0., 0.]] if same else bootstrap_gain(
        parent["sse"], candidate["sse"], parent["count"])
    point = np.sqrt(parent["sse"].sum(0)/parent["count"].sum(0))-np.sqrt(
        candidate["sse"].sum(0)/candidate["count"].sum(0))
    return {**common, "status": "verified", "terrains": len(parent["names"]),
            "ordered_terrain_names_and_counts_equal": True,
            "point_gain_selected12_minus_C148_dB": point.tolist(),
            "gain_selected12_minus_C148_dB_95CI": interval,
            "reused_parent_without_new_inference": same,
            "source": str(candidate_path.relative_to(ROOT)),
            "source_sha256": file_hash(candidate_path),
            "parent_source": str(parent_path.relative_to(ROOT)),
            "parent_source_sha256": file_hash(parent_path),
            "C82_interval_reference": {"source": evidence["source"],
                "json_field": "panels.monitor.paired_terrain_bootstrap.gain_C82_minus_candidate_dB_95CI"}}


def exposure_strata_diagnostic(evidence, model_digest, expected_row, config):
    """Posthoc cached-statistic diagnostic; both C82 exposure strata are development data."""
    common = {"selection_use": False, "posthoc_diagnostic": True,
              "scope_limit": "Both C82 beta-fit terrains and C82 model-selection terrains are development data; the 48 model-selection terrains are not a blind test.",
              "grouping_source": "audit/c79_receiver/protocol.json",
              "grouping_field": "panel[].terrain and panel[].group (fit/select/confirm)",
              "large_error_definition": "abs(original TRAIN label-frozen C82) >= 5 dB",
              "original_good_definition": "abs(original TRAIN label-frozen C82) <= 1 dB"}
    if evidence["status"] != "verified":
        return {**common, "status": "pending_independent_prediction_verification"}
    grouping_path = ROOT / common["grouping_source"]
    if not grouping_path.exists():
        return {**common, "status": "unrecorded", "missing_fields": [common["grouping_source"]]}
    grouping = load(grouping_path)
    if not isinstance(grouping.get("panel"), list) or any(
            not isinstance(row, dict) or "terrain" not in row or "group" not in row
            for row in grouping["panel"]):
        return {**common, "status": "unrecorded", "missing_fields": ["panel[].terrain/group"]}
    parent_path = PARENT / "selected_terrain_sse.npz"
    candidate_path = parent_path if model_digest == load(AUDIT / "protocol.json")["anchor_model_sha256"] else RUN / "selected_terrain_sse.npz"
    parent = saved_monitor_statistics(parent_path)
    candidate = saved_monitor_statistics(candidate_path)
    missing = {label: data["missing_map_fields"] for label, data in (("parent", parent), ("C148", candidate))
               if data["missing_map_fields"]}
    if missing:
        return {**common, "status": "unrecorded", "missing_fields": missing}
    for name in ("names", "map_ids", "map_terrain_index", "group_names", "map_group_count"):
        if not np.array_equal(parent[name], candidate[name]):
            raise ValueError(f"Parent/C148 exposure diagnostic identity differs: {name}")
    if not np.array_equal(candidate["map_ids"], np.asarray(config["panels"]["monitor"], dtype=np.int64)):
        raise ValueError("Exposure diagnostic ordered map IDs differ from frozen monitor")
    names = candidate["names"]
    indices = candidate["map_terrain_index"]
    if indices.ndim != 1 or len(indices) != len(candidate["map_ids"]) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Invalid saved map-to-terrain index")
    if np.any(indices < 0) or np.any(indices >= len(names)) or len(set(names.tolist())) != len(names):
        raise ValueError("Invalid saved monitor terrain identities")
    group_names = candidate["group_names"].tolist()
    if len(set(group_names)) != len(group_names):
        raise ValueError("Duplicate saved error-group names")
    missing_groups = [name for name in GROUPS.values() if name not in group_names]
    if missing_groups:
        return {**common, "status": "unrecorded", "missing_fields": missing_groups}
    expected_shape = (len(group_names), len(indices), 2)
    for data in (parent, candidate):
        for key in ("map_group_sse", "map_group_baseline_sse", "map_group_count"):
            array = data[key]
            if array.shape != expected_shape or not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError(f"Invalid cached exposure statistic: {key}")
        if not np.array_equal(data["map_group_count"], np.floor(data["map_group_count"])):
            raise ValueError("Noninteger valid pixel count")
    if not np.allclose(parent["map_group_baseline_sse"], candidate["map_group_baseline_sse"], atol=1e-5, rtol=1e-12):
        raise ValueError("Parent/C148 cached map C82 SSE differs")
    # Reconcile all requested group totals with already independent full-panel scores.
    parent_verified = load(PARENT / "selected_verified.json")["panels"]["monitor"]["groups"]
    for short, saved in GROUPS.items():
        index = group_names.index(saved)
        expected = group_metrics(expected_row["evaluation"]["monitor"]["groups"][short])
        for data, reference in ((candidate, expected), (parent, parent_verified[saved])):
            count = data["map_group_count"][index].sum(0)
            sse = data["map_group_sse"][index].sum(0)
            if not np.array_equal(count, reference["valid_pixels"]) or not np.allclose(
                    np.sqrt(sse/count), reference["RMSE_dB"], atol=1e-9, rtol=0):
                raise ValueError(f"Cached map group differs from independent score: {short}")
    membership = {}
    for row in grouping["panel"]:
        terrain, group = row["terrain"], row["group"]
        if terrain in membership and membership[terrain] != group:
            raise ValueError(f"C79 terrain has conflicting exposure groups: {terrain}")
        membership[terrain] = group
    absent = [str(name) for name in names if name not in membership]
    if absent:
        return {**common, "status": "unrecorded", "missing_terrain_group_assignments": absent}
    terrain_group = np.asarray([membership[name] for name in names])
    unexpected = sorted(set(terrain_group.tolist())-{"fit", "select"})
    if unexpected:
        raise ValueError(f"Current C145 monitor has unexpected C79 groups: {unexpected}")
    strata = {}
    for group, label in (("fit", "C82_beta_fit"), ("select", "C82_model_selection")):
        terrain_mask = terrain_group == group
        map_mask = terrain_mask[indices]
        metrics = {}
        for short, saved in GROUPS.items():
            index = group_names.index(saved)
            count = candidate["map_group_count"][index, map_mask].sum(0)
            if np.any(count <= 0):
                metrics[short] = {"status": "unrecorded", "reason": "No valid pixels for one frequency",
                                  "valid_pixels": count.astype(np.int64).tolist()}
                continue
            new_sse = candidate["map_group_sse"][index, map_mask].sum(0)
            old_sse = parent["map_group_sse"][index, map_mask].sum(0)
            base_sse = candidate["map_group_baseline_sse"][index, map_mask].sum(0)
            current, old, base = (np.sqrt(sse/count) for sse in (new_sse, old_sse, base_sse))
            metrics[short] = {"valid_pixels": count.astype(np.int64).tolist(),
                "C148_RMSE_dB": current.tolist(), "selected12_RMSE_dB": old.tolist(),
                "C82_RMSE_dB": base.tolist(), "gain_selected12_minus_C148_dB": (old-current).tolist(),
                "gain_C82_minus_C148_dB": (base-current).tolist(),
                "C148_better_than_selected12_both_bands": bool(np.all(current < old))}
        strata[label] = {"source_group": group, "terrains": int(terrain_mask.sum()),
                         "maps": int(map_mask.sum()), "groups": metrics}
    if [(strata[key]["terrains"], strata[key]["maps"]) for key in ("C82_beta_fit", "C82_model_selection")] != [(80, 480), (48, 148)]:
        raise ValueError("C79 exposure subdivision does not reproduce documented 80/480 + 48/148 scope")
    return {**common, "status": "verified", "strata": strata,
            "selected_model_sha256": model_digest,
            "reused_parent_without_new_inference": candidate_path == parent_path,
            "grouping_source_sha256": file_hash(grouping_path),
            "source": str(candidate_path.relative_to(ROOT)), "source_sha256": file_hash(candidate_path),
            "parent_source": str(parent_path.relative_to(ROOT)), "parent_source_sha256": file_hash(parent_path)}


def checkpoint_digest(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)["state"]
    value = hashlib.sha256()
    for name, tensor in state.items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def summary():
    plan, config, result = context()
    history = load(RUN / "history.json")
    historical = {row["absolute_epoch"]: row for row in load(PARENT / "history.json")}
    rows = {row["absolute_epoch"]: row for row in history}
    if len(rows) != len(history):
        raise ValueError("Duplicate absolute epochs in C148 history")
    first = plan["parent_absolute_epoch"]
    final = first + plan["additional_epochs"]
    expected_epochs = list(range(first + 4, final + 1, 4))
    if any(epoch not in rows or epoch not in historical for epoch in expected_epochs):
        raise ValueError("Missing matched full-panel evaluation")
    if result["absolute_epoch"] != final or max(rows) != final:
        raise ValueError("Fixed budget and final history epoch differ")
    parent = historical[first]
    selected_epoch = result["selected_absolute_epoch"]
    selected = rows[selected_epoch]
    selection_pool = [rows[epoch] for epoch in [first, *expected_epochs]]
    mean = lambda row: float(np.mean(row["evaluation"]["monitor"]["rmse"]))
    chosen = min(selection_pool, key=mean)
    if chosen["absolute_epoch"] != selected_epoch:
        raise ValueError("Selected checkpoint is not original joint-mean selection")
    if not np.allclose(result["selected_monitor_RMSE_dB"], selected["evaluation"]["monitor"]["rmse"], atol=1e-12, rtol=0):
        raise ValueError("Training result selected metric differs")
    selected_model = checkpoint_digest(RUN / CHECKPOINTS["selected"])
    selected_is_parent = selected_model == plan["anchor_model_sha256"]
    if selected_is_parent != result["selected_remains_parent"]:
        raise ValueError("Selected model identity and training result disagree")
    selected_metrics = row_metrics(selected)
    selected_vs_parent = compare_rows(selected, parent)
    improvements = selected_vs_parent["monitor"]["all"]["C148_better_in_both_bands"] and all(
        gain > 0 for gain in selected_metrics["monitor"]["all"]["gain_C82_minus_candidate_dB"])
    if bool(improvements) != result["improves_both_monitor_bands_over_parent_and_C82"]:
        raise ValueError("Frozen joint-band success condition disagrees with result")
    evidence = verified_evidence("selected", selected_model, selected, config)
    uncertainty = parent_gain_uncertainty(evidence, selected_model, selected)
    exposure = exposure_strata_diagnostic(evidence, selected_model, selected, config)
    matched = []
    lr_differs = []
    used_lr_unrecorded = []
    used_lr_differs = []
    for epoch in expected_epochs:
        candidate, control = rows[epoch], historical[epoch]
        same_lr = math.isclose(candidate["learning_rate"], control["learning_rate"], rel_tol=0, abs_tol=1e-15)
        if not same_lr:
            lr_differs.append(epoch)
        used_candidate = candidate.get("learning_rate_used")
        used_control = control.get("learning_rate_used")
        same_used_lr = None
        if used_candidate is None or used_control is None:
            used_lr_unrecorded.append(epoch)
        else:
            same_used_lr = math.isclose(used_candidate, used_control, rel_tol=0, abs_tol=1e-15)
            if not same_used_lr:
                used_lr_differs.append(epoch)
        matched.append({"absolute_epoch": epoch, "C148": row_metrics(candidate),
                        "historical_C145": row_metrics(control),
                        "gain": compare_rows(candidate, control),
                        "post_evaluation_learning_rate_C148": candidate["learning_rate"],
                        "post_evaluation_learning_rate_C145": control["learning_rate"],
                        "same_post_evaluation_learning_rate": same_lr,
                        "recorded_training_learning_rate_used_C148": used_candidate,
                        "recorded_training_learning_rate_used_C145": used_control,
                        "same_recorded_training_learning_rate_used": same_used_lr,
                        "relative_parameter_distance": candidate["relative_parameter_distance"],
                        "anchor_penalty": candidate["anchor_penalty"]})
    output = {
        "status": "verified" if evidence["status"] == "verified" else "awaiting_independent_prediction_verification",
        "campaign": "C148_parameter_anchor", "frequency_hz": [415_000_000, 5_775_000_000],
        "protocol_sha256": file_hash(AUDIT / "protocol.json"),
        "report_source_sha256": file_hash(Path(__file__)), "config_sha256": file_hash(RUN / "config.json"),
        "history_sha256": file_hash(RUN / "history.json"),
        "historical_control_history_sha256": file_hash(PARENT / "history.json"),
        "control_training_repeated": False, "new_training_epochs": plan["additional_epochs"],
        "official_VAL_used": False, "scope": plan["evaluation_scope"],
        "large_error_definition": "abs(original TRAIN label - frozen development C82) >= 5 dB",
        "original_good_definition": "abs(original TRAIN label - frozen development C82) <= 1 dB",
        "selection_rule": "one same checkpoint minimizing joint mean monitor RMSE; no per-band cherry-picking",
        "selected_absolute_epoch": selected_epoch, "selected_model_sha256": selected_model,
        "selected_checkpoint_sha256": file_hash(RUN / CHECKPOINTS["selected"]),
        "selected_remains_parent": selected_is_parent, "selected": selected_metrics,
        "selected_vs_selected12": selected_vs_parent,
        "same_checkpoint_beats_selected12_and_C82_both_bands": bool(improvements),
        "independent_prediction_evidence": evidence,
        "paired_terrain_gain_over_selected12": uncertainty,
        "posthoc_C82_exposure_strata": exposure,
        "new_method_gain_verified": bool(improvements and evidence["status"] == "verified"),
        "parent_selected12": row_metrics(parent), "last": row_metrics(rows[final]),
        "last_vs_historical_C145_same_epoch": compare_rows(rows[final], historical[final]),
        "epoch_matched": matched,
        "post_evaluation_learning_rate_divergence_epochs": lr_differs,
        "recorded_training_learning_rate_divergence_epochs": used_lr_differs,
        "recorded_training_learning_rate_comparison_unavailable_epochs": used_lr_unrecorded,
        "learning_rate_semantics": "learning_rate in both histories is after scheduler.step at evaluation (for the next interval). C148 separately records learning_rate_used for the evaluated epoch; historical C145 does not, so missing used-LR comparisons remain unavailable.",
        "limitations": [
            "Single predeclared parameter-anchor arm and seed; no sweep or automatic extension.",
            "Repeated development monitor, not a new blind test; C82 calibrator previously saw 80/128 monitor terrains.",
            "Reducing late overfitting or retaining the parent is not a new-model improvement.",
            "Original scheduler rule is preserved; post-evaluation LR differences are listed, while absent historical training-used LR remains unrecorded.",
            "Paired terrain bootstrap is a final diagnostic and does not adjust for checkpoint selection or repeated development-set use.",
            "The posthoc C82 beta-fit80/model-selection48 exposure subdivision uses the same selected checkpoint; both subdivisions remain development data and are not used for new selection."
        ]}
    atomic_json(AUDIT / "comparison_summary.json", output)
    print(json.dumps({"status": output["status"], "selected_absolute_epoch": selected_epoch,
                      "selected_remains_parent": selected_is_parent,
                      "selected_monitor_RMSE_dB": selected_metrics["monitor"]["all"]["RMSE_dB"],
                      "last_monitor_RMSE_dB": output["last"]["monitor"]["all"]["RMSE_dB"],
                      "same_checkpoint_beats_selected12_and_C82_both_bands": bool(improvements),
                      "new_method_gain_verified": output["new_method_gain_verified"],
                      "independent_prediction_evidence": evidence,
                      "gain_over_selected12_dB_95CI": uncertainty.get("gain_selected12_minus_C148_dB_95CI"),
                      "summary": str((AUDIT / "comparison_summary.json").relative_to(ROOT))}, ensure_ascii=False), flush=True)


def export(tags):
    plan, config, _ = context()
    uuid = plan["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid or os.environ.get("CC_GPU_UUID") != uuid:
        raise ValueError("Export requires the protocol full UUID in both CUDA_VISIBLE_DEVICES and CC_GPU_UUID")
    from src.cc_core import gpu_guard, check_source_unchanged
    from src.c105_context import ResidualPanel, digest, reader
    from src.c123_lossless import create
    from src.c145_train import evaluate, prediction_identity, cached_prediction_metadata, record_prediction
    import torch

    check_source_unchanged()
    completed = []
    pending = []
    for tag in tags:
        checkpoint = RUN / CHECKPOINTS[tag]
        model = checkpoint_digest(checkpoint)
        if model == plan["anchor_model_sha256"]:
            # Existing CPU verification is authoritative; avoid recreating predictions.
            parent_verified = load(PARENT / "selected_verified.json")
            if parent_verified["export_identity"]["model_sha256"] != model:
                raise ValueError("Parent verification identity differs")
            completed.append({"tag": tag, "status": "reuse_verified_parent",
                              "verified_source": "runs/c145_capacity_native/selected_verified.json",
                              "GPU_inference_performed": False})
        else:
            pending.append((tag, checkpoint, model))
    if not pending:
        print(json.dumps(completed), flush=True)
        return
    gpu_guard()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    data = reader()
    panels = {name: ResidualPanel(data, config["panels"][name], "context")
              for name in ("fit", "monitor")}
    net = create("native").cuda().to(memory_format=torch.channels_last)
    batch = load(ROOT / plan["base_protocol"])["batch"]
    exported_models = {}
    for tag, path, model in pending:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["c148_protocol_sha256"] != config["protocol_sha256"]:
            raise ValueError("Export checkpoint protocol differs")
        net.load_state_dict(checkpoint["state"], strict=True)
        if digest(net) != model:
            raise ValueError("GPU-restored export model identity differs")
        metadata = {"tag": tag, "checkpoint": path.name, "model_sha256": model,
                    "training_state": checkpoint["training_state"], "prediction_dtype": "float32",
                    "forward_precision": "BF16 autocast", "panels": {}}
        for name, panel in panels.items():
            identity = prediction_identity(model, panel)
            destination = RUN / f"{tag}_{name}_prediction.npy"
            cache = cached_prediction_metadata(destination, identity)
            if cache is None:
                previous = exported_models.get(model)
                previous_path = RUN / f"{previous}_{name}_prediction.npy" if previous else None
                previous_cache = cached_prediction_metadata(previous_path, identity) if previous_path else None
                temporary = destination.with_suffix(".partial.npy")
                if temporary.exists():
                    temporary.unlink()
                if previous_cache:
                    os.link(previous_path, temporary)
                    temporary.replace(destination)
                    cache = record_prediction(destination, identity, linked_metadata=previous_cache,
                                              linked_from=str(previous_path.relative_to(RUN)))
                else:
                    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                                     shape=tuple(identity["shape"]))
                    evaluate(net, panel, batch, f"C148 export {tag} {name}", predictions=array)
                    del array
                    temporary.replace(destination)
                    cache = record_prediction(destination, identity)
            metadata["panels"][name] = {"file": destination.name, **cache}
        atomic_json(RUN / f"{tag}_export.json", metadata)
        exported_models.setdefault(model, tag)
        completed.append({"tag": tag, "status": "exported_or_reused_cache",
                          "export": str((RUN / f"{tag}_export.json").relative_to(ROOT)),
                          "next": f"CUDA_VISIBLE_DEVICES='' .venv/bin/python -m src.c145_verify --run runs/c148_parameter_anchor --tag {tag}"})
    print(json.dumps(completed), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("summary", help="CPU-only matched comparison after complete training")
    command = subs.add_parser("export", help="Explicit GPU export only for new C148 checkpoint(s)")
    command.add_argument("--tags", nargs="+", choices=tuple(CHECKPOINTS), default=["selected"])
    args = parser.parse_args()
    if args.command == "summary":
        summary()
    else:
        export(list(dict.fromkeys(args.tags)))


if __name__ == "__main__":
    main()
