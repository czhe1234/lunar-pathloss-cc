"""Shared C151 paired-report runner; new candidates, fixed development references.

CPU-only. Reuses immutable C150 score/statistic helpers without changing their
module globals or any earlier run. Supports matched A/B candidate directories
and optional already-verified historical controls.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.c150_report import (ARMS, PANELS, GROUPS, TRAIN_GROUPS, BANDS, FREQUENCIES,
    SEED, REPLICATES, PARENT, atomic, clean, load, sha, paired_delta, statistics,
    evidence, align, parent_evidence, posthoc_regions)


def checked_protocol(path, campaign):
    plan = load(path)
    if plan.get("campaign") != campaign:
        raise ValueError("Campaign name differs from the selected protocol")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen stage source changed: " + name)
    old = load(ROOT / "audit/c145_capacity/protocol.json")
    if plan["panels"] != old["panels"]:
        raise ValueError("Exact ordered C145 FIT/monitor split required")
    if (plan.get("epochs"), plan.get("updates_per_arm"), plan.get("batch")) != (12, 4224, 8):
        raise ValueError("C151 matched fixed budget must be12epochs/4224updates/batch8")
    if plan.get("official_VAL_used"):
        raise ValueError("Official VAL may not enter this comparison")
    if set(plan["arms"]) != set(ARMS):
        raise ValueError("Paired stage must declare exactly A and B")
    return plan


def run_context(directory, plan=None, protocol_sha=None, arm=None, campaign=None):
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    config, result = load(directory / "config.json"), load(directory / "result.json")
    history, exported = load(directory / "history.json"), load(directory / "selected_export.json")
    own_plan = config["protocol"]
    if (result.get("status") != "complete" or result.get("epochs") != 12
            or result.get("updates") != 4224 or result.get("official_VAL_used")):
        raise ValueError("Fixed candidate/control run is not complete")
    if config["protocol_sha256"] != result["protocol_sha256"]:
        raise ValueError("Run config and result protocol identity differ")
    if plan is not None:
        if (config["panels"] != plan["panels"] or own_plan != plan
                or config["protocol_sha256"] != protocol_sha
                or config.get("arm") != arm or result.get("arm") != arm
                or config.get("campaign") != campaign):
            raise ValueError("Candidate stage/config identity differs")
    else:
        # Historical controls must bring independently verified evidence; they
        # do not rerun training, inference, or their old raw-data verifier.
        if config["panels"] != own_plan["panels"]:
            raise ValueError("Historical config/panel mismatch")
    epochs = [row["epoch"] for row in history]
    if epochs != [0, 4, 8, 12]:
        raise ValueError("Exactly the matched0/4/8/12 evaluation opportunities required")
    if any(row["updates"] != row["epoch"]*352 for row in history):
        raise ValueError("Recorded epoch/update relationship differs")
    selected = min(history, key=lambda x: float(np.mean(x["evaluation"]["monitor"]["rmse"])))
    if selected["epoch"] != result["selected_epoch"]:
        raise ValueError("Selected predictor is not the first joint-mean minimum")
    if not np.allclose(selected["evaluation"]["monitor"]["rmse"],
                       result["selected_monitor_RMSE_dB"], atol=1e-12, rtol=0):
        raise ValueError("Selected score differs from training result")
    if (exported["checkpoint"] != "selected.pt"
            or exported["checkpoint_sha256"] != sha(directory / "selected.pt")
            or exported["training_state"]["epoch"] != selected["epoch"]
            or exported["model_sha256"] != selected["predictor_sha256"]
            or exported["model_sha256"] != result["selected_predictor_sha256"]
            or exported["net_state_sha256"] != selected["net_state_sha256"]):
        raise ValueError("Full selected predictor/export/checkpoint identity differs")
    return {"directory": directory, "config": config, "result": result,
            "history": history, "selected": selected, "export": exported,
            "arm": arm if arm is not None else config["arm"]}


def verify_contexts(contexts):
    """Verify disjoint new saved predictions on two CPU workers, then join.

    The verifier outputs belong to their respective candidate directories.
    Shared original labels/masks/baselines are read only. Reporting and SwanLab
    writes occur only after every verifier and its evidence checks succeed.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

    def one(arm, context):
        if evidence(context, hash_predictions=True) is not None:
            print(json.dumps({"arm": arm, "independent_verification": "existing_valid_evidence_reused",
                              "network_inference": False}), flush=True)
            return
        print(json.dumps({"arm": arm, "independent_verification": "CPU_scoring_started",
                          "network_inference": False}), flush=True)
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "src.c145_verify",
                        "--run", str(context["directory"].relative_to(ROOT)),
                        "--tag", "selected"], cwd=ROOT, env=env, check=True)
        if evidence(context) is None:
            raise ValueError("Independent verifier did not produce complete evidence: " + arm)
        print(json.dumps({"arm": arm, "independent_verification": "CPU_scoring_passed",
                          "network_inference": False}), flush=True)

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(one, arm, context): arm for arm, context in contexts.items()}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as error:
                errors.append((pending[future], error))
    if errors:
        raise RuntimeError("Candidate verification failed: " + "; ".join(
            arm + ": " + repr(error) for arm, error in errors)) from errors[0][1]


def selected_metrics(verified):
    return {panel: {short: {key: verified["panels"][panel]["groups"][long][key]
                             for key in ("RMSE_dB", "SSE_dB2", "valid_pixels",
                                         "C82_RMSE_dB", "C82_SSE_dB2")}
                    for short, long in zip(TRAIN_GROUPS, GROUPS)}
            for panel in PANELS}


def tracking(summary, destination, previous, enabled):
    identity = hashlib.sha256(json.dumps(clean(summary), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (previous and previous.get("tracking", {}).get("status") == "complete"
            and previous["tracking"].get("analysis_payload_sha256") == identity):
        return previous["tracking"]
    if not enabled:
        return {"status": "not_requested", "reason": "Explicit --no-swanlab"}
    from src import cc_tracking
    run = cc_tracking.init(name=summary["campaign"] + " verified paired comparison",
        campaign=summary["campaign"],
        config={"run_kind": "analysis.paired_comparison",
                "analysis_payload_sha256": identity,
                "protocol_sha256": summary["protocol_sha256"],
                "summary": str(destination.relative_to(ROOT)),
                "training_sources": {arm: summary["arms"][arm]["tracking"] for arm in ARMS},
                "_cc_dashboard": {"kind": "analysis",
                                  "scope": "saved-prediction paired model comparison; no training"}},
        description="Matched candidate A/B and cached performance references; local CPU evidence only.",
        tags=["analysis", "paired-terrain", "no-new-training"])
    run_info = {"run_id": run.id, "dir": str(run.dir)}
    metrics = {}
    for arm in ARMS:
        metrics[f"gate/{arm}_passed"] = int(summary["arms"][arm]["gate_passed"])
        for reference, result in summary["arms"][arm]["paired_comparisons"].items():
            for channel, band in enumerate(BANDS):
                prefix = f"paired/{arm}/{reference}/{band}"
                metrics[prefix + "/delta_RMSE_dB"] = result["point_delta_RMSE_dB"][channel]
                metrics[prefix + "/CI95_lower_dB"] = result["delta_RMSE_dB_95CI"][channel][0]
                metrics[prefix + "/CI95_upper_dB"] = result["delta_RMSE_dB_95CI"][channel][1]
    try:
        cc_tracking.log(metrics, step=0)
    finally:
        cc_tracking.finish()
    return {"status": "complete", **run_info, "analysis_payload_sha256": identity, "mode": "local"}


def make_summary(plan, contexts, protocol_path, campaign, historical_dirs,
                 destination, log_swanlab=True):
    parent_verified = parent_evidence()
    parent = statistics(PARENT / "selected_terrain_sse.npz", plan["panels"], parent_verified)
    verified, states = {}, {}
    for arm in ARMS:
        verified[arm] = evidence(contexts[arm], hash_predictions=True)
        if verified[arm] is None:
            raise ValueError("Both candidates need independent saved-prediction verification")
        states[arm] = statistics(contexts[arm]["directory"] / "selected_terrain_sse.npz",
                                 plan["panels"], verified[arm])
        align(states[arm], parent)
    # Both matched interventions must start with the same analytic output prior.
    for panel in PANELS:
        x = contexts["A"]["history"][0]["evaluation"][panel]
        y = contexts["B"]["history"][0]["evaluation"][panel]
        if not np.allclose(x["rmse"], y["rmse"], atol=1e-12, rtol=0):
            raise ValueError("Matched A/B initial analytic predictions differ")
    historical = {}
    for path in historical_dirs:
        context = run_context(path)
        value = evidence(context, hash_predictions=True)
        if value is None:
            raise ValueError("Historical control is not independently verified; do not rerun it here")
        if context["config"]["panels"] != plan["panels"]:
            raise ValueError("Historical control uses different panels")
        state = statistics(context["directory"] / "selected_terrain_sse.npz", plan["panels"], value)
        align(state, parent)
        label = str(context["directory"].relative_to(ROOT))
        historical[label] = {"state": state, "context": context, "verified": value}
    output = {
        "status": "verified", "campaign": campaign,
        "protocol": str(protocol_path.relative_to(ROOT)), "protocol_sha256": sha(protocol_path),
        "report_source_sha256": sha(Path(__file__)),
        "shared_helper_source_sha256": sha(ROOT / "src/c150_report.py"),
        "frequency_hz": list(FREQUENCIES), "scope": plan["scope"],
        "gate_definition": plan["gate"], "official_VAL_used": False,
        "training_or_network_inference_repeated": False, "arms": {},
        "references": {"C145_selected12": {
            "verified_json_sha256": sha(PARENT / "selected_verified.json"),
            "statistics_sha256": sha(PARENT / "selected_terrain_sse.npz"),
            "model_sha256": parent_verified["export_identity"]["model_sha256"]}},
        "controlled_A_B": {
            "labels": plan["arms"],
            "question": plan.get("question"),
            "factor": plan.get("controlled_difference", plan.get("changed_factor")),
            "initial_analytic_predictions_equal": True,
            "scope": "B-minus-A tests the declared matched intervention only. Historical model comparisons are performance references with possible architecture/input differences."},
        "interpretation_limits": [
            "C82 and C145 selected12 comparisons are performance gates, not one-factor tests of adding diffraction features.",
            "Historical C145 may use C82 context/initialization absent from these teacher-free models.",
            "For the C150 pair, only the analytic output prior differs; it does not isolate the benefit of adding diffraction inputs.",
            "No comparison identifies missing original generator inputs or proves solver-floor censoring caused the original residuals.",
            "CIs condition on repeated monitor use and selected checkpoints; no blind or selection-corrected claim.",
            "Initial analytic checkpoint may win; one same joint-mean checkpoint is used for both bands."]}
    for arm in ARMS:
        context, st = contexts[arm], states[arm]["monitor"]
        comparisons = {
            "C82": paired_delta(st["terrain_sse"], parent["monitor"]["terrain_C82_sse"], st["terrain_count"]),
            "C145_selected12": paired_delta(st["terrain_sse"], parent["monitor"]["terrain_sse"], st["terrain_count"])}
        historical_comparisons = {}
        for label, control in historical.items():
            historical_comparisons[label] = paired_delta(
                st["terrain_sse"], control["state"]["monitor"]["terrain_sse"], st["terrain_count"])
        directory = context["directory"]
        output["arms"][arm] = {
            "label": plan["arms"][arm], "selected_epoch": context["selected"]["epoch"],
            "selected_initial_analytic_checkpoint": context["selected"]["epoch"] == 0,
            "predictor_sha256": context["export"]["model_sha256"],
            "net_state_sha256": context["export"]["net_state_sha256"],
            "selected": selected_metrics(verified[arm]), "paired_comparisons": comparisons,
            "historical_performance_comparisons": historical_comparisons,
            "gate_passed": all(v["both_band_upper_CI_strictly_below_zero"] for v in comparisons.values()),
            "last_original_training_evaluation": context["history"][-1]["evaluation"],
            "last_independently_verified": context["history"][-1]["predictor_sha256"] == context["export"]["model_sha256"],
            "tracking": load(directory / "tracking.json"),
            "evidence": {"run_directory": str(directory.relative_to(ROOT)),
                "verified_json_sha256": sha(directory / "selected_verified.json"),
                "statistics_sha256": sha(directory / "selected_terrain_sse.npz"),
                "selected_export_sha256": sha(directory / "selected_export.json"),
                "config_sha256": sha(directory / "config.json"), "history_sha256": sha(directory / "history.json")}}
    output["matched_B_minus_A"] = paired_delta(states["B"]["monitor"]["terrain_sse"],
                                               states["A"]["monitor"]["terrain_sse"],
                                               states["A"]["monitor"]["terrain_count"])
    output["matched_control_improvement_resolved_both_bands"] = output["matched_B_minus_A"]["both_band_upper_CI_strictly_below_zero"]
    output["historical_controls"] = {label: {
        "campaign": item["context"]["config"]["campaign"],
        "selected_epoch": item["context"]["selected"]["epoch"],
        "predictor_sha256": item["context"]["export"]["model_sha256"],
        "verified_json_sha256": sha(item["context"]["directory"] / "selected_verified.json"),
        "statistics_sha256": sha(item["context"]["directory"] / "selected_terrain_sse.npz")}
        for label, item in historical.items()}
    output["posthoc_C149_regions"] = posthoc_regions(plan, contexts, verified)
    output["budget_scale_allowed"] = any(output["arms"][arm]["gate_passed"] for arm in ARMS)
    output["passing_arms"] = [arm for arm in ARMS if output["arms"][arm]["gate_passed"]]
    output["lowest_development_joint_mean_arm"] = min(
        ARMS, key=lambda arm: np.mean(output["arms"][arm]["selected"]["monitor"]["all"]["RMSE_dB"]))
    output = clean(output)
    previous = load(destination) if destination.exists() else None
    output["tracking"] = tracking(output, destination, previous, log_swanlab)
    atomic(destination, output)
    destination.with_suffix(".md").write_text(markdown(output))
    print(json.dumps({"status": "verified", "campaign": campaign,
        "selected": {arm: {"epoch": output["arms"][arm]["selected_epoch"],
                          "RMSE_dB": output["arms"][arm]["selected"]["monitor"]["all"]["RMSE_dB"],
                          "gate": output["arms"][arm]["gate_passed"]} for arm in ARMS},
        "matched_B_minus_A": output["matched_B_minus_A"],
        "summary": str(destination.relative_to(ROOT))}, ensure_ascii=False), flush=True)


def markdown(summary):
    pair = lambda values: " / ".join("未定义" if x is None else f"{x:.6f}" for x in values)
    lines = [f"# {summary['campaign']} 受控对照", "",
             str(summary["controlled_A_B"]["question"]), "",
             "A/B归因只对应声明的受控差异；历史模型仅作性能参照。两频采用同一选中 checkpoint。",
             "此处是重复使用的开发 monitor，不是官方 VAL 或盲测。差值为 candidate − reference，负值表示改善。", "",
             "| 版本 | 条件 | Epoch | FIT RMSE 415 / 5.775 GHz (dB) | monitor RMSE 415 / 5.775 GHz (dB) | 门槛 |",
             "|---|---|---:|---:|---:|---|"]
    for arm in ARMS:
        value = summary["arms"][arm]
        lines.append(f"| {arm} | {value['label']} | {value['selected_epoch']} | "
                     f"{pair(value['selected']['fit']['all']['RMSE_dB'])} | "
                     f"{pair(value['selected']['monitor']['all']['RMSE_dB'])} | "
                     f"{'通过' if value['gate_passed'] else '未通过'} |")
    lines += ["", "## 双参照配对地形95% CI", "",
              "| 版本 | 参照 | ΔRMSE 415 / 5.775 GHz (dB) | 415 MHz95% CI | 5.775 GHz95% CI |",
              "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        value = summary["arms"][arm]
        merged = {**value["paired_comparisons"], **value["historical_performance_comparisons"]}
        for reference, comparison in merged.items():
            ci = comparison["delta_RMSE_dB_95CI"]
            lines.append(f"| {arm} | {reference} | {pair(comparison['point_delta_RMSE_dB'])} | "
                         f"[{ci[0][0]:.6f}, {ci[0][1]:.6f}] | [{ci[1][0]:.6f}, {ci[1][1]:.6f}] |")
    contrast = summary["matched_B_minus_A"]
    lines += ["", "## 匹配A/B差异", "",
              "B − A 双频差值 (dB)："+pair(contrast["point_delta_RMSE_dB"])+"。",
              "两频区间均严格小于0："+("是。" if summary["matched_control_improvement_resolved_both_bands"] else "否。"),
              "这不是历史盲骨干与新骨干的单变量比较。", "",
              "## 完整FIT / monitor及固定难区", "",
              "| 版本 | 面板 | 区域 | RMSE 415 / 5.775 GHz (dB) |", "|---|---|---|---:|"]
    for arm in ARMS:
        for panel in PANELS:
            for group in ("all", "tail", "C82_good"):
                lines.append(f"| {arm} | {panel} | {group} | "
                             f"{pair(summary['arms'][arm]['selected'][panel][group]['RMSE_dB'])} |")
    lines += ["", "## C149固定Fresnel分组（事后诊断）", "",
              "| 版本 | 区域 | RMSE 415 / 5.775 GHz (dB) | 总MSE贡献 415 / 5.775 GHz (%) |",
              "|---|---|---:|---:|"]
    for arm in ARMS:
        for region in ("Grazing", "Shallow_Shadow"):
            value = summary["posthoc_C149_regions"]["arms"][arm][region]["candidate"]
            lines.append(f"| {arm} | {region} | {pair(value['RMSE_dB'])} | "
                         f"{pair(value['share_of_total_SSE_percent'])} |")
    lines += ["", "## 范围与判断", "",
              "允许扩大预算："+("是。" if summary["budget_scale_allowed"] else "否。"),
              *["- "+note for note in summary["interpretation_limits"]], "",
              "selected由保存预测独立CPU FP64计分；其他last仅引用原训练评估。"]
    return "\n".join(lines)+"\n"


def preflight():
    parent = parent_evidence()
    panels = load(ROOT / "audit/c145_capacity/protocol.json")["panels"]
    state = statistics(PARENT / "selected_terrain_sse.npz", panels, parent)
    zero = paired_delta(state["monitor"]["terrain_sse"], state["monitor"]["terrain_sse"],
                        state["monitor"]["terrain_count"])
    assert zero["delta_RMSE_dB_95CI"] == [[0., 0.], [0., 0.]]
    assert not zero["both_band_upper_CI_strictly_below_zero"]
    assert "torch" not in sys.modules
    print(json.dumps({"status": "CPU_generic_report_preflight_passed",
        "paired_terrain_count": len(state["monitor"]["terrain_names"]),
        "shared_helper_sha256": sha(ROOT / "src/c150_report.py"),
        "source_sha256": sha(Path(__file__)), "GPU_used": False}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("preflight", "verify", "summary", "all"),
                        default="summary")
    parser.add_argument("--protocol", type=Path, default=Path("audit/c151_global/protocol.json"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/c151_global"))
    parser.add_argument("--campaign", default="C151a")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-a", type=Path)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--historical-control", type=Path, action="append", default=[])
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return
    absolute = lambda path: path if path.is_absolute() else ROOT / path
    protocol_path = absolute(args.protocol)
    plan = checked_protocol(protocol_path, args.campaign)
    run_root = absolute(args.run_root)
    paths = {"A": absolute(args.run_a) if args.run_a else run_root / "A",
             "B": absolute(args.run_b) if args.run_b else run_root / "B"}
    contexts = {arm: run_context(path, plan, sha(protocol_path), arm, args.campaign)
                for arm, path in paths.items()}
    if args.command in ("verify", "all"):
        verify_contexts(contexts)
    if args.command in ("summary", "all"):
        directory = absolute(args.output_dir) if args.output_dir else protocol_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        make_summary(plan, contexts, protocol_path, args.campaign,
                     [absolute(path) for path in args.historical_control],
                     directory / "comparison_summary.json", not args.no_swanlab)


if __name__ == "__main__":
    main()
