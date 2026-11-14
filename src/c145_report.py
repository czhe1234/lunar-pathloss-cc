"""Build C145's Chinese capability report only from complete, identity-checked results.

CPU only: this module and c145_verify import NumPy and the standard library.
Missing or inconsistent evidence never creates either final report.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import c145_verify as verifier

TAGS = ("selected", "fit_best", "last")
PANELS = ("fit", "monitor")
CHECKPOINTS = {"selected": "best_monitor.pt", "fit_best": "best_fit.pt", "last": "last.pt"}
TRAIN_GROUPS = {
    "all": "all",
    "C82_abs_error_ge_5dB": "tail",
    "C82_underestimated_loss_ge_5dB": "under_tail",
    "C82_overestimated_loss_ge_5dB": "over_tail",
    "C82_abs_error_lt_5dB": "non_tail",
    "C82_abs_error_le_1dB": "C82_good",
}
STOP_REASON = "weighted_full_FIT_plateau_after_settled_learning_rate_reductions"
SUMMARY_PATH = ROOT / "audit/c145_capacity/comparison_summary.json"
REPORT_PATH = ROOT / "results/C145_充分训练能力对照_20261112.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pair(value, name, *, nonnegative=False, integer=False):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (2,) and np.isfinite(array).all(), f"{name}: expected two finite bands")
    if nonnegative:
        require(np.all(array >= 0), f"{name}: negative value")
    if integer:
        require(np.array_equal(array, array.astype(np.int64)), f"{name}: noninteger count")
    return array


def close(actual, expected, name, *, atol=1e-6, rtol=1e-10):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), f"{name}: values disagree")


class Ledger:
    """Keep read identities stable and hash hard-linked prediction bytes only once."""
    def __init__(self):
        self.identities = {}
        self.hashes = {}
        self.paths = {}

    def before(self, path):
        path = Path(path).resolve()
        identity = verifier.file_identity(path)
        previous = self.identities.setdefault(path, identity)
        require(previous == identity, f"Input changed: {path}")
        return path, identity

    def sha256(self, path):
        path, identity = self.before(path)
        if identity not in self.hashes:
            self.hashes[identity] = verifier.file_hash(path)
        self.unchanged(path)
        self.paths[str(path)] = self.hashes[identity]
        return self.hashes[identity]

    def read_json(self, path):
        path, identity = self.before(path)
        raw = path.read_bytes()
        self.unchanged(path)
        digest = hashlib.sha256(raw).hexdigest()
        previous = self.hashes.setdefault(identity, digest)
        require(previous == digest, f"Input bytes changed: {path}")
        self.paths[str(path)] = digest
        return json.loads(raw)

    def unchanged(self, path=None):
        items = self.identities.items() if path is None else (
            (Path(path).resolve(), self.identities[Path(path).resolve()]),
        )
        for item, identity in items:
            require(verifier.file_identity(item) == identity, f"Input changed: {item}")


def validate_training(config, plan, initial, history, result):
    require(config.get("official_VAL_used") is False, "config must exclude official VAL")
    require(result.get("official_VAL_used") is False, "training_result must exclude official VAL")
    noncausal = {"created", "source_sha256", "initial_phase_additional_epochs",
                 "extension_epochs_if_still_learning"}
    for key, value in plan.items():
        if key not in noncausal:
            require(config.get(key) == value, f"Frozen C145 configuration differs: {key}")
    state = result["training_state"]
    require(result.get("status") == "training_complete", "Training has not reached its stopping condition")
    epoch = state["additional_epoch"]
    steps_per_epoch = (len(config["panels"]["fit"]) + config["batch"] - 1) // config["batch"]
    require(isinstance(epoch, int) and epoch > 0, "Final epoch must be positive")
    require(state["additional_step"] == epoch * steps_per_epoch, "Final step/epoch mismatch")
    require(result["phase_end_epoch"] == epoch, "Final phase epoch mismatch")
    require(state.get("stop_reason") == STOP_REASON, "Unexpected or absent training stop reason")
    stop_checks = {
        "minimum_additional_epochs": epoch >= config["minimum_additional_epochs"],
        "minimum_lr_reductions": state["learning_rate_reductions"] >= config["minimum_lr_reductions_for_stop"],
        "FIT_plateau_epochs": epoch - state["last_meaningful_fit_improvement_epoch"] >= config["plateau_epochs"],
        "trained_since_last_LR_reduction": epoch - state["last_lr_reduction_epoch"] >= config["epochs_after_last_lr_reduction"],
    }
    require(all(stop_checks.values()), "Recorded training stop does not satisfy the frozen conditions")
    require(isinstance(history, list) and len(history) > 0, "Full training evaluation history is missing")
    previous_epoch = 0
    best_fit = float(initial["fit"]["weighted_MSE_objective"])
    best_monitor = float(np.mean(pair(initial["monitor"]["rmse"], "initial monitor RMSE")))
    require(np.isfinite(best_fit) and best_fit >= 0, "Invalid initial FIT objective")
    best_fit_epoch = best_monitor_epoch = 0
    meaningful_best, last_meaningful = best_fit, 0
    reductions, last_reduction = 0, 0
    previous_rate = float(config["learning_rate"])
    for record in history:
        current_epoch = record["additional_epoch"]
        require(isinstance(current_epoch, int) and previous_epoch < current_epoch <= epoch,
                "History epochs are not strictly increasing")
        require(record["absolute_epoch"] == current_epoch + config["parent_epoch"],
                "History absolute epoch mismatch")
        require(record["additional_step"] == current_epoch * steps_per_epoch,
                "History step/epoch mismatch")
        objective = float(record["evaluation"]["fit"]["weighted_MSE_objective"])
        monitor = float(np.mean(pair(record["evaluation"]["monitor"]["rmse"], "history monitor RMSE")))
        require(np.isfinite(objective) and objective >= 0, "Invalid history FIT objective")
        if objective < best_fit:
            best_fit, best_fit_epoch = objective, current_epoch
        if monitor < best_monitor:
            best_monitor, best_monitor_epoch = monitor, current_epoch
        if objective < meaningful_best * (1 - config["minimum_relative_fit_improvement"]):
            meaningful_best, last_meaningful = objective, current_epoch
        rate = float(record["learning_rate"])
        require(np.isfinite(rate) and rate > 0 and rate <= previous_rate * (1 + 1e-8),
                "Invalid evaluation learning-rate history")
        if rate < previous_rate * (1 - 1e-8):
            reductions += 1
            last_reduction = current_epoch
            close(rate, max(previous_rate * config["lr_factor"], config["minimum_learning_rate"]),
                  "LR reduction factor", atol=1e-12, rtol=1e-8)
        require(record["learning_rate_reductions"] == reductions, "History LR-reduction count mismatch")
        require(record["last_lr_reduction_epoch"] == last_reduction, "History last LR-reduction epoch mismatch")
        require(record["epochs_since_last_lr_reduction"] == current_epoch - last_reduction,
                "History settled-LR duration mismatch")
        require(record["epochs_since_meaningful_FIT_improvement"] == current_epoch - last_meaningful,
                "History meaningful-FIT plateau duration mismatch")
        require(record["best_fit_epoch"] == best_fit_epoch and record["best_monitor_epoch"] == best_monitor_epoch,
                "History best checkpoint selection mismatch")
        require(record.get("stop_reason") is None or current_epoch == epoch,
                "Training continued beyond a recorded stopping point")
        previous_epoch, previous_rate = current_epoch, rate
    require(previous_epoch == epoch, "Final epoch has no full evaluation")
    for key in ("additional_epoch", "additional_step", "learning_rate_reductions",
                "last_lr_reduction_epoch", "best_fit_epoch", "best_monitor_epoch", "stop_reason"):
        require(history[-1][key] == state[key], f"Final history/training-state mismatch: {key}")
    require(state["last_meaningful_fit_improvement_epoch"] == last_meaningful,
            "Final meaningful FIT-improvement epoch mismatch")
    close(state["best_fit_objective"], best_fit, "Best FIT objective", atol=1e-10)
    close(state["best_monitor_mean_rmse"], best_monitor, "Best monitor objective", atol=1e-10)
    close(state["meaningful_fit_best"], meaningful_best, "Meaningful FIT objective", atol=1e-10)
    return {
        **state,
        "parent_epoch": config["parent_epoch"],
        "absolute_epoch": epoch + config["parent_epoch"],
        "steps_per_epoch": steps_per_epoch,
        "evaluation_records": len(history),
        "final_learning_rate": previous_rate,
        "stop_checks": stop_checks,
        "stopping_requirements": {
            key: config[key] for key in (
                "minimum_additional_epochs", "minimum_lr_reductions_for_stop",
                "minimum_relative_fit_improvement", "plateau_epochs", "epochs_after_last_lr_reduction")
        },
        "FIT_plateau_epochs_actual": epoch - last_meaningful,
        "epochs_after_last_LR_reduction_actual": epoch - last_reduction,
        "initial_weighted_FIT_objective": initial["fit"]["weighted_MSE_objective"],
        "final_weighted_FIT_objective": history[-1]["evaluation"]["fit"]["weighted_MSE_objective"],
    }


def validate_export(run, tag, config, verified, ledger):
    export_path = run / f"{tag}_export.json"
    exported = ledger.read_json(export_path)
    require(exported.get("tag") == tag, f"{tag}: wrong export tag")
    require(exported.get("checkpoint") == CHECKPOINTS[tag], f"{tag}: wrong selected checkpoint")
    model = exported["model_sha256"]
    require(isinstance(model, str) and len(model) == 64 and set(model) <= set("0123456789abcdef"),
            f"{tag}: invalid model digest")
    identity = verified["export_identity"]
    for key in ("model_sha256", "checkpoint", "prediction_dtype", "forward_precision"):
        require(identity.get(key) == exported.get(key), f"{tag}: verified/export {key} mismatch")
    require(identity["export_record_sha256"] == ledger.sha256(export_path),
            f"{tag}: completed export digest mismatch")
    require(exported.get("prediction_dtype") == "float32" and
            exported.get("forward_precision") == "BF16 autocast", f"{tag}: precision metadata mismatch")
    for panel in PANELS:
        ids = config["panels"][panel]
        path = run / f"{tag}_{panel}_prediction.npy"
        sidecar_path = path.with_suffix(".json")
        recorded = ledger.read_json(sidecar_path)
        completed = exported["panels"][panel]
        require(completed.get("file") == path.name, f"{tag}/{panel}: prediction filename mismatch")
        require({key: value for key, value in completed.items() if key != "file"} == recorded,
                f"{tag}/{panel}: incomplete or mismatched export sidecar")
        expected = verifier.panel_identity(model, ids)
        for key, value in expected.items():
            require(recorded.get(key) == value, f"{tag}/{panel}: metadata mismatch: {key}")
        _, file_id = ledger.before(path)
        require(recorded["file_size_bytes"] == file_id[2], f"{tag}/{panel}: file-size mismatch")
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)
        require(list(prediction.shape) == expected["shape"] and prediction.dtype == np.float32,
                f"{tag}/{panel}: prediction shape/dtype mismatch")
        del prediction
        prediction_hash = ledger.sha256(path)
        require(recorded["prediction_sha256"] == prediction_hash, f"{tag}/{panel}: prediction SHA256 mismatch")
        score = verified["panels"][panel]
        require(score["prediction_sha256"] == prediction_hash, f"{tag}/{panel}: stale verified prediction")
        require(score["prediction_sidecar_sha256"] == ledger.sha256(sidecar_path),
                f"{tag}/{panel}: stale verified sidecar")
        require(score["ordered_panel_ids_sha256"] == expected["panel_ids_sha256"],
                f"{tag}/{panel}: stale verified panel ordering")
        require(Path(score["prediction_path"]).resolve() == path.resolve(),
                f"{tag}/{panel}: verified prediction path mismatch")
    return exported


def validate_npz(run, tag, verified, config, panel_terrains, ledger):
    path = run / f"{tag}_terrain_sse.npz"
    ledger.sha256(path)
    with np.load(path, allow_pickle=False) as arrays:
        require(tuple(arrays["group_names"].tolist()) == verifier.GROUPS, f"{tag}: NPZ group ordering mismatch")
        for panel in PANELS:
            summary = verified["panels"][panel]
            ids = config["panels"][panel]
            names = arrays[f"{panel}_terrain_names"]
            terrain_index = arrays[f"{panel}_map_terrain_index"]
            require(np.array_equal(arrays[f"{panel}_map_ids"], ids), f"{tag}/{panel}: NPZ map ordering mismatch")
            require(names.tolist() == sorted(set(panel_terrains[panel])), f"{tag}/{panel}: NPZ terrain names mismatch")
            expected_index = np.asarray([names.tolist().index(name) for name in panel_terrains[panel]])
            require(np.array_equal(terrain_index, expected_index), f"{tag}/{panel}: NPZ terrain mapping mismatch")
            statistics = {}
            for key in verifier.STAT_NAMES:
                values = arrays[f"{panel}_map_group_{key}"]
                require(values.shape == (len(verifier.GROUPS), len(ids), 2) and np.isfinite(values).all(),
                        f"{tag}/{panel}: NPZ statistic shape/nonfinite: {key}")
                statistics[key] = values
            for index, group in enumerate(verifier.GROUPS):
                computed = verifier.summarize({key: values[index].sum(0) for key, values in statistics.items()})
                recorded = summary["groups"][group]
                for key, values in computed.items():
                    close(recorded[key], values, f"{tag}/{panel}/{group}/{key}", atol=1e-5, rtol=1e-9)
                for key in ("valid_pixels", "improved_pixels", "worsened_pixels", "correct_correction_direction_pixels"):
                    pair(recorded[key], f"{tag}/{panel}/{group}/{key}", nonnegative=True, integer=True)
            for key in ("sse", "baseline_sse", "count"):
                values = statistics[key]
                close(values[1] + values[4], values[0], f"{tag}/{panel}: tail partition {key}")
                close(values[2] + values[3], values[1], f"{tag}/{panel}: signed-tail partition {key}")
                close(np.maximum(values[5] - values[4], 0), 0, f"{tag}/{panel}: good-region subset {key}")
            for field, key in (("sse", "sse"), ("C82_sse", "baseline_sse"), ("count", "count")):
                total = np.zeros((len(names), 2))
                np.add.at(total, terrain_index, statistics[key][0])
                close(arrays[f"{panel}_terrain_{field}"], total, f"{tag}/{panel}: terrain totals {field}")
            new_errors = summary["new_errors"]
            for key in ("C82_le_1_to_error_gt_3", "C82_le_1_to_error_gt_5", "C82_ge_5_to_error_lt_1"):
                pair(new_errors[key], f"{tag}/{panel}/{key}", nonnegative=True, integer=True)
            require(np.all(np.asarray(new_errors["C82_le_1_to_error_gt_5"]) <= new_errors["C82_le_1_to_error_gt_3"]),
                    f"{tag}/{panel}: new-error thresholds are inconsistent")
            require(np.all(np.asarray(new_errors["C82_le_1_to_error_gt_3"]) <= summary["groups"]["C82_abs_error_le_1dB"]["valid_pixels"]),
                    f"{tag}/{panel}: new-error count exceeds original good region")
            require(np.all(np.asarray(new_errors["C82_ge_5_to_error_lt_1"]) <= summary["groups"]["C82_abs_error_ge_5dB"]["valid_pixels"]),
                    f"{tag}/{panel}: repaired-tail count exceeds fixed tail")
            if panel == "monitor" and tag == "selected":
                bootstrap = verifier.paired_terrain_bootstrap(
                    arrays["monitor_terrain_sse"], arrays["monitor_terrain_C82_sse"], arrays["monitor_terrain_count"])
                recorded = summary["paired_terrain_bootstrap"]
                for key in ("unit", "terrains", "seed", "replicates", "method_limit"):
                    require(recorded[key] == bootstrap[key], f"selected monitor CI metadata differs: {key}")
                close(recorded["gain_C82_minus_candidate_dB_95CI"],
                      bootstrap["gain_C82_minus_candidate_dB_95CI"], "selected monitor paired-terrain CI", atol=1e-12)
    ledger.unchanged(path)


def validate_against_evaluation(tag, verified, exported, initial, history, state):
    epoch_key = {"selected": "best_monitor_epoch", "fit_best": "best_fit_epoch", "last": "additional_epoch"}[tag]
    epoch = state[epoch_key]
    require(exported["training_state"]["additional_epoch"] == epoch, f"{tag}: exported checkpoint epoch mismatch")
    require(exported["training_state"]["additional_step"] == epoch * state["steps_per_epoch"],
            f"{tag}: exported checkpoint step mismatch")
    reference = initial if epoch == 0 else next(
        (record["evaluation"] for record in history if record["additional_epoch"] == epoch), None)
    require(reference is not None, f"{tag}: selected epoch has no actual evaluation")
    for panel in PANELS:
        for group, train_group in TRAIN_GROUPS.items():
            recorded = reference[panel]["groups"][train_group]
            independently_scored = verified["panels"][panel]["groups"][group]
            require(recorded["count"] == independently_scored["valid_pixels"], f"{tag}/{panel}/{group}: evaluation count mismatch")
            close(independently_scored["RMSE_dB"], recorded["RMSE"], f"{tag}/{panel}/{group}: evaluation RMSE", atol=2e-5)
            close(independently_scored["C82_RMSE_dB"], recorded["C82_RMSE"], f"{tag}/{panel}/{group}: C82 evaluation RMSE", atol=2e-6)
        for cutoff in (3, 5):
            require(reference[panel][f"C82_good_new_error_above_{cutoff}dB_count"] ==
                    verified["panels"][panel]["new_errors"][f"C82_le_1_to_error_gt_{cutoff}"],
                    f"{tag}/{panel}: evaluation new-error count mismatch")
    if epoch == 0:
        require(exported["model_sha256"] == state["initial_model_state_sha256"],
                f"{tag}: epoch-zero model differs from the starting checkpoint")
    return {"additional_epoch": epoch, "absolute_epoch": epoch + state["parent_epoch"],
            "model_sha256": exported["model_sha256"], "checkpoint": exported["checkpoint"]}


def load_evidence(run):
    ledger = Ledger()
    required = [run / filename for filename in ("config.json", "initial.json", "history.json", "training_result.json")]
    for tag in TAGS:
        required.extend(run / f"{tag}_{suffix}" for suffix in
                        ("verified.json", "terrain_sse.npz", "export.json",
                         "fit_prediction.npy", "monitor_prediction.npy",
                         "fit_prediction.json", "monitor_prediction.json"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        shown = "; ".join(missing[:4])
        remainder = f"；另缺 {len(missing) - 4} 项" if len(missing) > 4 else ""
        raise FileNotFoundError(f"真实训练/导出/独立评分尚未齐全；缺少 {len(missing)} 项：{shown}{remainder}")
    config = ledger.read_json(run / "config.json")
    initial = ledger.read_json(run / "initial.json")
    history = ledger.read_json(run / "history.json")
    training_result = ledger.read_json(run / "training_result.json")
    plan = ledger.read_json(ROOT / "audit/c145_capacity/protocol.json")
    frozen = ledger.read_json(ROOT / "audit/c105_context/protocol.json")
    for panel in PANELS:
        ids = config["panels"][panel]
        require(len(ids) == len(set(ids)) and set(ids) == set(frozen[panel]), f"{panel}: fixed C105 panel mismatch")
    require(len(config["panels"]["fit"]) == 2815 and len(config["panels"]["monitor"]) == 628,
            "Unexpected development-panel map counts")
    index_path = ROOT.parent / "data/train_index.csv"
    ledger.before(index_path)
    with index_path.open() as stream:
        rows = list(csv.DictReader(stream))
    ledger.sha256(index_path)
    panel_terrains = {panel: [rows[index]["terrain_id"] for index in config["panels"][panel]] for panel in PANELS}
    require(not set(panel_terrains["fit"]) & set(panel_terrains["monitor"]), "FIT/monitor terrains overlap")
    actual_panels = {panel: {"maps": len(config["panels"][panel]), "terrains": len(set(panel_terrains[panel]))}
                     for panel in PANELS}
    require(actual_panels == {"fit": {"maps": 2815, "terrains": 1984}, "monitor": {"maps": 628, "terrains": 128}},
            "Unexpected development-panel terrain counts")
    require(config["actual_panels"] == actual_panels, "Recorded panel sizes differ from original TRAIN index")
    require(config["parameters"] == 15214786, "Unexpected trainable branch parameter count")
    training = validate_training(config, plan, initial, history, training_result)
    training["initial_model_state_sha256"] = config["initial_model_state_sha256"]
    common_hashes = {
        "verification_source_sha256": ledger.sha256(ROOT / "src/c145_verify.py"),
        "config_sha256": ledger.sha256(run / "config.json"),
        "frozen_protocol_sha256": ledger.sha256(ROOT / "audit/c105_context/protocol.json"),
        "baseline_fit_index_sha256": ledger.sha256(ROOT / "audit/c101_residual/base0of1/indices.json"),
        "baseline_monitor_index_sha256": ledger.sha256(ROOT / "audit/c88_mesh/base_index.json"),
    }
    results, checkpoint_epochs = {}, {}
    for tag in TAGS:
        verified = ledger.read_json(run / f"{tag}_verified.json")
        require(verified.get("status") == "verified" and verified.get("tag") == tag, f"{tag}: incomplete independent score")
        require(verified.get("official_VAL_used") is False, f"{tag}: official VAL must be excluded")
        require(verified["frequency_hz"] == [415000000, 5775000000], f"{tag}: frequency mismatch")
        for key, expected in common_hashes.items():
            require(verified.get(key) == expected, f"{tag}: stale verification identity: {key}")
        for panel in PANELS:
            for key, expected in actual_panels[panel].items():
                require(verified["panels"][panel][key] == expected, f"{tag}/{panel}: wrong {key} count")
        exported = validate_export(run, tag, config, verified, ledger)
        validate_npz(run, tag, verified, config, panel_terrains, ledger)
        checkpoint_epochs[tag] = validate_against_evaluation(tag, verified, exported, initial, history, training)
        results[tag] = verified
    for panel in PANELS:
        for group in verifier.GROUPS:
            reference = results["selected"]["panels"][panel]["groups"][group]
            for tag in TAGS:
                metrics = results[tag]["panels"][panel]["groups"][group]
                for key in ("C82_RMSE_dB", "C82_SSE_dB2", "valid_pixels"):
                    close(metrics[key], reference[key], f"{tag}/{panel}/{group}: frozen C82 denominator")
                close(initial[panel]["groups"][TRAIN_GROUPS[group]]["C82_RMSE"], reference["C82_RMSE_dB"],
                      f"initial/{panel}/{group}: same C82 baseline", atol=2e-6)
                require(initial[panel]["groups"][TRAIN_GROUPS[group]]["count"] == reference["valid_pixels"],
                        f"initial/{panel}/{group}: same raw valid-pixel masks")
    monitor = results["selected"]["panels"]["monitor"]["groups"]["all"]
    require(monitor["valid_pixels"] == [40864973, 37837148], "Full monitor denominator differs")
    close(monitor["C82_RMSE_dB"], [2.1463905073492664, 2.4180144849255294],
          "Full monitor C82 reference", atol=2e-6, rtol=0)
    ledger.sha256(ROOT / "audit/c145_capacity/protocol.json")
    ledger.sha256(Path(__file__))
    ledger.unchanged()
    return config, initial, history, training, results, checkpoint_epochs, ledger


def build_summary(config, initial, history, training, results, checkpoints, provenance):
    comparisons = {}
    for tag in TAGS:
        comparisons[tag] = {}
        for panel in PANELS:
            comparisons[tag][panel] = {}
            for group, train_group in TRAIN_GROUPS.items():
                score = results[tag]["panels"][panel]["groups"][group]
                initial_rmse = np.asarray(initial[panel]["groups"][train_group]["RMSE"])
                candidate_rmse = np.asarray(score["RMSE_dB"])
                comparisons[tag][panel][group] = {
                    "gain_C82_minus_candidate_dB": (np.asarray(score["C82_RMSE_dB"]) - candidate_rmse).tolist(),
                    "gain_epoch8_initial_minus_candidate_dB": (initial_rmse - candidate_rmse).tolist(),
                    "net_SSE_removed_dB2": score["SSE_removed_dB2"],
                }
    selected_gain = np.asarray(comparisons["selected"]["monitor"]["all"]["gain_C82_minus_candidate_dB"])
    fit_uniform_tags = [
        tag for tag in TAGS
        if all(np.all(np.asarray(comparisons[tag]["fit"][group]["gain_C82_minus_candidate_dB"]) > 0)
               for group in ("all", "C82_abs_error_ge_5dB"))
    ]
    if np.all(selected_gain > 0):
        category = "observed_both_band_monitor_gain"
        conclusion = "selected 在该 monitor 上双频 RMSE 都下降。下降幅度按实测值报告，小幅下降也属于观测收益。"
    elif fit_uniform_tags:
        category = "FIT_and_tail_gain_without_joint_monitor_gain"
        conclusion = "已有权重在完整 FIT 和固定尾区双频都下降，但 selected 未在 monitor 上双频同时下降。训练内收益尚未完整转移到该 monitor。"
    else:
        category = "no_joint_full_FIT_and_tail_gain"
        conclusion = "三个已保存权重尚未同时做到完整 FIT 和固定尾区双频都下降。单频或局部收益仍按实测值保留。"
    interpretation = {
        "class": category,
        "conclusion": conclusion,
        "FIT_both_bands_and_tail_improved_tags": fit_uniform_tags,
        "selected_monitor_gain_dB": selected_gain.tolist(),
        "capability_and_promotion_are_separate": True,
        "automatic_full_TRAIN_expansion": False,
        "scope_limit": "Only this branch, continuation optimizer/loss and reused development panels; no global capacity sufficiency or missing-original-input conclusion.",
    }
    trajectory = [{
        "additional_epoch": 0, "absolute_epoch": config["parent_epoch"],
        "fit_RMSE_dB": initial["fit"]["rmse"], "monitor_RMSE_dB": initial["monitor"]["rmse"],
        "weighted_FIT_objective": initial["fit"]["weighted_MSE_objective"],
    }]
    trajectory.extend({
        "additional_epoch": record["additional_epoch"], "absolute_epoch": record["absolute_epoch"],
        "fit_RMSE_dB": record["evaluation"]["fit"]["rmse"],
        "monitor_RMSE_dB": record["evaluation"]["monitor"]["rmse"],
        "weighted_FIT_objective": record["evaluation"]["fit"]["weighted_MSE_objective"],
        "learning_rate": record["learning_rate"],
        "learning_rate_reductions": record["learning_rate_reductions"],
    } for record in history)
    return {
        "status": "complete_identity_checked",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": config["question"], "model": config["model"], "parameters": config["parameters"],
        "frequency_hz": [415000000, 5775000000],
        "panel_scope": {
            **config["actual_panels"],
            "FIT_is_official_full_TRAIN": False, "monitor_is_fresh_blind": False,
            "monitor_terrains_previously_seen_by_C82_calibrator": 80,
            "official_VAL_used": False,
        },
        "training": training, "checkpoint_epochs": checkpoints,
        "initial_epoch8": initial, "trajectory": trajectory,
        "comparisons": comparisons, "verified_results": results,
        "interpretation": interpretation,
        "continuation_limits": {
            "fresh_AdamW": True, "whole_FIT_denominator_correction": True,
            "pure_extra_epochs_causal_comparison": False,
            "monitor_selection_optimism": True,
        },
        "source_sha256": provenance,
    }


def dual(values, precision=6, signed=False, integer=False):
    if integer:
        return " / ".join(f"{int(value):,}" for value in values)
    specification = ("+" if signed else "") + f".{precision}f"
    return " / ".join(format(float(value), specification) for value in values)


def render_report(summary):
    results, train = summary["verified_results"], summary["training"]
    stop = train["stopping_requirements"]
    lines = [
        "# C145 充分训练能力对照",
        "",
        summary["interpretation"]["conclusion"],
        "",
        "模型是冻结的开发版 C82，加 C123 的 15,214,786 参数原生残差分支。"
        "本次从 C123 第 8 轮权重继续训练。三个最终权重的误差由保存的 Float32 预测、原始 TRAIN 标签和原始掩码用 NumPy FP64 独立计算。前向使用 BF16。"
        "起点数值来自 initial.json；已核对其掩码计数和 C82 分母。",
        "",
        "表内双频顺序为 415 MHz / 5.775 GHz。RMSE 越低越好。"
        "净 SSE 收益 = C82 SSE − 当前 SSE；正数表示总体平方误差减少。"
        "尾区按 C82 的原始绝对误差 ≥5 dB 固定，所有权重使用同一批像素。",
        "",
        "完整 FIT 有 2,815 张图、1,984 个地形。它是完整开发 FIT，**不是官方全部 TRAIN**。"
        "monitor 有 628 张图、128 个地形，与 FIT 地形不重叠。它已被反复用于开发；其中 80 个地形曾参与 C82 校准。"
        "这里的“新地形”仅指相对本分支 FIT 的地形留出，不能视为全流程全新盲测。未用官方 VAL。",
        "",
    ]
    tables = (
        ("完整 FIT 与新地形 monitor", "all"),
        ("固定大误差尾区（C82 绝对误差 ≥5 dB）", "C82_abs_error_ge_5dB"),
        ("非尾区（C82 绝对误差 <5 dB）", "C82_abs_error_lt_5dB"),
        ("原本预测较准的区域（C82 绝对误差 ≤1 dB）", "C82_abs_error_le_1dB"),
    )
    for title, group in tables:
        lines.extend([title, "",
                      "| 权重 | FIT RMSE (dB) | FIT 净 SSE 收益 (dB²) | monitor RMSE (dB) | monitor 净 SSE 收益 (dB²) |",
                      "|---|---:|---:|---:|---:|"])
        fit_ref = results["selected"]["panels"]["fit"]["groups"][group]
        monitor_ref = results["selected"]["panels"]["monitor"]["groups"][group]
        lines.append(f"| C82 | {dual(fit_ref['C82_RMSE_dB'])} | 0 / 0 | {dual(monitor_ref['C82_RMSE_dB'])} | 0 / 0 |")
        for tag in TAGS:
            fit = results[tag]["panels"]["fit"]["groups"][group]
            monitor = results[tag]["panels"]["monitor"]["groups"][group]
            lines.append(f"| {tag} | {dual(fit['RMSE_dB'])} | {dual(fit['SSE_removed_dB2'], 2, signed=True)} | "
                         f"{dual(monitor['RMSE_dB'])} | {dual(monitor['SSE_removed_dB2'], 2, signed=True)} |")
        lines.extend(["", f"固定有效像素：FIT {dual(fit_ref['valid_pixels'], integer=True)}；"
                          f"monitor {dual(monitor_ref['valid_pixels'], integer=True)}。", ""])
    lines.extend([
        "selected 的正负尾区检查。低估损耗指 C82 的预测路径损耗偏低 ≥5 dB；高估损耗反之。",
        "",
        "| 面板 / 尾区 | C82 RMSE (dB) | selected RMSE (dB) | 净 SSE 收益 (dB²) | 改正方向正确的像素比例 |",
        "|---|---:|---:|---:|---:|",
    ])
    for panel in PANELS:
        for label, group in (("低估损耗", "C82_underestimated_loss_ge_5dB"),
                             ("高估损耗", "C82_overestimated_loss_ge_5dB")):
            score = results["selected"]["panels"][panel]["groups"][group]
            lines.append(f"| {panel} / {label} | {dual(score['C82_RMSE_dB'])} | {dual(score['RMSE_dB'])} | "
                         f"{dual(score['SSE_removed_dB2'], 2, signed=True)} | "
                         + " / ".join(f"{value * 100:.2f}%" for value in score["correct_correction_direction_fraction"]) + " |")
    lines.extend([
        "", "新增错误与修复。新增错误的起点均为 C82 原本误差 ≤1 dB 的同一像素。C82 自身新增错误为 0。",
        "",
        "| 权重 / 面板 | 新增误差 >3 dB（像素） | 新增误差 >5 dB（像素） | 固定尾区降到 <1 dB（像素） |",
        "|---|---:|---:|---:|",
    ])
    for tag in TAGS:
        for panel in PANELS:
            errors = results[tag]["panels"][panel]["new_errors"]
            lines.append(f"| {tag} / {panel} | {dual(errors['C82_le_1_to_error_gt_3'], integer=True)} | "
                         f"{dual(errors['C82_le_1_to_error_gt_5'], integer=True)} | "
                         f"{dual(errors['C82_ge_5_to_error_lt_1'], integer=True)} |")
    lines.extend([
        "", "实际训练与权重选择。selected 按 monitor 双频平均 RMSE 选取；"
        "fit_best 按完整 FIT 的加权 MSE 选取；last 是实际停止轮。三个权重全部报告。",
        "",
        "| 权重 | 新增训练轮 | 含起点的总轮数 |",
        "|---|---:|---:|",
    ])
    for tag in TAGS:
        checkpoint = summary["checkpoint_epochs"][tag]
        lines.append(f"| {tag} | {checkpoint['additional_epoch']} | {checkpoint['absolute_epoch']} |")
    lines.extend([
        "",
        f"实际新增 {train['additional_epoch']} 轮，共 {train['additional_step']:,} 次参数更新；"
        f"含 C123 起点共 {train['absolute_epoch']} 轮。学习率下降 {train['learning_rate_reductions']} 次，"
        f"最终学习率 {train['final_learning_rate']:.8g}。完整评估记录 {train['evaluation_records']} 次。",
        f"停止时，完整 FIT 已连续 {train['FIT_plateau_epochs_actual']} 轮没有达到规定的相对改善；"
        f"最近一次降学习率后已实际训练 {train['epochs_after_last_LR_reduction_actual']} 轮。"
        f"停止要求为：新增至少 {stop['minimum_additional_epochs']} 轮；学习率至少下降 "
        f"{stop['minimum_lr_reductions_for_stop']} 次；加权 FIT MSE 连续至少 {stop['plateau_epochs']} 轮"
        f"没有达到 {100 * stop['minimum_relative_fit_improvement']:.3g}% 的相对改善；"
        f"最后一次降学习率后至少再训练 {stop['epochs_after_last_lr_reduction']} 轮。monitor 不决定停止。",
        f"停止标记：`{train['stop_reason']}`。",
        "",
        "| 比较 | 初始 C123 第 8 轮 RMSE (dB) | selected RMSE 降幅 (dB) | fit_best RMSE 降幅 (dB) | last RMSE 降幅 (dB) |",
        "|---|---:|---:|---:|---:|",
    ])
    for panel in PANELS:
        for label, group in (("全部", "all"), ("固定尾区", "C82_abs_error_ge_5dB")):
            initial = summary["initial_epoch8"][panel]["groups"][TRAIN_GROUPS[group]]["RMSE"]
            gains = [dual(summary["comparisons"][tag][panel][group]["gain_epoch8_initial_minus_candidate_dB"], signed=True)
                     for tag in TAGS]
            lines.append(f"| {panel} / {label} | {dual(initial)} | " + " | ".join(gains) + " |")
    ci = results["selected"]["panels"]["monitor"]["paired_terrain_bootstrap"]
    intervals = ci["gain_C82_minus_candidate_dB_95CI"]
    gains = summary["comparisons"]["selected"]["monitor"]["all"]["gain_C82_minus_candidate_dB"]
    lines.extend([
        "", f"selected 在 monitor 上相对 C82 的 RMSE 降幅为 {dual(gains, signed=True)} dB。"
        f"按地形配对重采样 {ci['replicates']:,} 次的 95% 区间为 "
        f"[{intervals[0][0]:+.6f}, {intervals[0][1]:+.6f}] / "
        f"[{intervals[1][0]:+.6f}, {intervals[1][1]:+.6f}] dB。"
        "每次保留同一地形下全部图和像素。区间只反映这个重复使用的开发面板；未校正选权重带来的乐观偏差。",
        "",
        "本次能力判断与后续扩训门槛分开。任何下降都按大小保留，"
        "没有达到某个扩训门槛不等于没有能力收益。本轮不自动进行官方全部 TRAIN 扩训。",
        "",
        "本次使用新 AdamW，并修正了训练损失的全 FIT 分母。"
        "因此，起点到终点是这套续训流程的能力对照，不能把变化全部归因于多训练了若干轮。",
        "若完整 FIT 和尾区仍无一致改善，需要继续区分优化、目标和本分支表达问题；"
        "若 FIT 改善而 monitor 未保留，要检查泛化和区域间的误差交换；"
        "若 monitor 双频改善，则已有该流程的开发收益证据，仍需保留原好区代价和面板复用限制。"
        "这些结果都不能单独证明模型容量已全局充分，也不能证明原始输入信息缺失。",
        "",
        "身份核验覆盖 run 配置、完整训练轨迹、三个导出的模型摘要、面板顺序、Float32 文件实际 SHA256、"
        "导出记录、独立评分源码摘要及地形统计。完整数字、逐频收益、全部轨迹和输入摘要见 "
        "`audit/c145_capacity/comparison_summary.json`。逐图/地形统计保存在 run 的三个 `*_terrain_sse.npz`。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/c145_capacity_native"))
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    try:
        config, initial, history, training, results, checkpoints, ledger = load_evidence(run)
        summary = build_summary(config, initial, history, training, results, checkpoints, dict(ledger.paths))
        report = render_report(summary)
        ledger.unchanged()
        # Neither destination is touched until every actual training/export/score check succeeds.
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        report_temporary = REPORT_PATH.with_suffix(".partial.md")
        report_temporary.write_text(report, encoding="utf-8")
        report_temporary.replace(REPORT_PATH)
        summary["report_path"] = str(REPORT_PATH)
        summary["report_sha256"] = verifier.file_hash(REPORT_PATH)
        verifier.atomic_json(SUMMARY_PATH, summary)
        print(json.dumps({"status": summary["status"], "summary": str(SUMMARY_PATH), "report": str(REPORT_PATH)},
                         ensure_ascii=False), flush=True)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        print(f"C145 报告未生成：{error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

