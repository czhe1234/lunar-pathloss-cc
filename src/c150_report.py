"""C150 independent saved-prediction verification and paired CPU report.

Never imports a model, executes network inference, or exports new predictions.
All differences are candidate RMSE minus reference RMSE; negative is better.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "audit/c150_diffraction_training"
RUNS = ROOT / "runs/c150_diffraction_prior"
PARENT = ROOT / "runs/c145_capacity_native"
C149 = ROOT / "audit/c149_error_audit"
ARMS = ("A", "B")
PANELS = ("fit", "monitor")
GROUPS = ("all", "C82_abs_error_ge_5dB", "C82_underestimated_loss_ge_5dB",
          "C82_overestimated_loss_ge_5dB", "C82_abs_error_lt_5dB",
          "C82_abs_error_le_1dB")
TRAIN_GROUPS = ("all", "tail", "under_tail", "over_tail", "non_tail", "C82_good")
FREQUENCIES = (415_000_000, 5_775_000_000)
BANDS = ("415MHz", "5p775GHz")
SEED = 145073
REPLICATES = 2000


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def ratio(numerator, denominator):
    n, d = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    return np.divide(n, d, out=np.full(np.broadcast_shapes(n.shape, d.shape), np.nan),
                     where=d > 0)


def paired_delta(candidate_sse, reference_sse, counts):
    """Terrain-paired percentile CI; one resample retains every pixel of a terrain."""
    current = np.asarray(candidate_sse, np.float64)
    old = np.asarray(reference_sse, np.float64)
    count = np.asarray(counts, np.int64)
    if current.shape != old.shape or current.shape != count.shape:
        raise ValueError("Paired statistic shapes differ")
    if current.ndim != 2 or current.shape[1] != 2 or np.any(count < 0):
        raise ValueError("Expected terrain x two-band sufficient statistics")
    if not np.isfinite(current).all() or not np.isfinite(old).all():
        raise ValueError("Nonfinite SSE")
    if np.any(current < 0) or np.any(old < 0) or np.any(count.sum(0) <= 0):
        raise ValueError("Invalid SSE/support")
    rng = np.random.default_rng(SEED)
    samples = rng.integers(len(current), size=(REPLICATES, len(current)))
    denominator = count[samples].sum(1)
    sampled = (np.sqrt(ratio(current[samples].sum(1), denominator))
               - np.sqrt(ratio(old[samples].sum(1), denominator)))
    if not np.isfinite(sampled).all():
        raise ValueError("Bootstrap produced a zero-support replicate")
    interval = np.quantile(sampled, [.025, .975], axis=0).T
    point = np.sqrt(ratio(current.sum(0), count.sum(0))) - np.sqrt(
        ratio(old.sum(0), count.sum(0)))
    return {"difference": "candidate_RMSE_minus_reference_RMSE_dB",
            "point_delta_RMSE_dB": point.tolist(),
            "delta_RMSE_dB_95CI": interval.tolist(),
            "both_band_upper_CI_strictly_below_zero": bool(np.all(interval[:, 1] < 0)),
            "unit": "terrain; all maps and valid pixels retained per sampled terrain",
            "terrains": len(current), "seed": SEED, "replicates": REPLICATES,
            "method_limit": "Conditional diagnostic on reused development monitor and selected checkpoints; not a selection-corrected or blind CI."}


def protocol():
    path = AUDIT / "protocol.json"
    plan = load(path)
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if sha(ROOT / plan["base_protocol"]) != plan["base_protocol_sha256"]:
        raise ValueError("C145 frozen panel protocol changed")
    base = load(ROOT / plan["base_protocol"])
    if plan["panels"] != base["panels"]:
        raise ValueError("Ordered fixed panels differ from C145")
    if (plan["epochs"], plan["updates_per_arm"], plan["batch"]) != (12, 4224, 8):
        raise ValueError("Predeclared C150 budget differs")
    if plan.get("official_VAL_used") or not plan["cache"].get("ready"):
        raise ValueError("Expected a complete input-only cache and no official VAL")
    return plan


def arm_context(arm, plan):
    directory = RUNS / arm
    config = load(directory / "config.json")
    result = load(directory / "result.json")
    history = load(directory / "history.json")
    exported = load(directory / "selected_export.json")
    identity = sha(AUDIT / "protocol.json")
    if config.get("arm") != arm or result.get("arm") != arm:
        raise ValueError("Arm identity differs")
    if (config["protocol_sha256"] != identity or result["protocol_sha256"] != identity
            or config["panels"] != plan["panels"]):
        raise ValueError("Arm protocol/panel identity differs")
    if (result.get("status") != "complete" or result["epochs"] != 12
            or result["updates"] != 4224 or result.get("official_VAL_used")):
        raise ValueError("Arm fixed training budget is not complete")
    if [row["epoch"] for row in history] != [0, 4, 8, 12]:
        raise ValueError("Missing or duplicate fixed-budget evaluations")
    if any(row["updates"] != row["epoch"] * 352 for row in history):
        raise ValueError("Recorded update count differs")
    chosen = min(history, key=lambda row: float(np.mean(row["evaluation"]["monitor"]["rmse"])))
    if chosen["epoch"] != result["selected_epoch"]:
        raise ValueError("Selected checkpoint is not the first joint-mean minimum")
    selected = next(row for row in history if row["epoch"] == result["selected_epoch"])
    if not np.allclose(selected["evaluation"]["monitor"]["rmse"],
                       result["selected_monitor_RMSE_dB"], atol=1e-12, rtol=0):
        raise ValueError("Selected metric and result differ")
    if (exported["model_sha256"] != result["selected_predictor_sha256"]
            or exported["model_sha256"] != selected["predictor_sha256"]
            or exported["net_state_sha256"] != selected["net_state_sha256"]
            or exported["training_state"]["epoch"] != selected["epoch"]
            or exported["checkpoint"] != "selected.pt"
            or exported["checkpoint_sha256"] != sha(directory / "selected.pt")):
        raise ValueError("Saved selected predictor/checkpoint identity differs")
    # Predictor identity includes the fixed arm prior and feature cache, not just net.
    definition = {"net_state_sha256": selected["net_state_sha256"], "arm": arm,
                  "prior": plan["arms"][arm],
                  "feature_cache_sha256": plan["cache"]["features_sha256"]}
    predictor = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    if predictor != selected["predictor_sha256"]:
        raise ValueError("Full predictor digest differs")
    return {"arm": arm, "directory": directory, "config": config, "result": result,
            "history": history, "selected": selected, "export": exported}


def evidence(context, hash_predictions=False):
    directory, exported = context["directory"], context["export"]
    path = directory / "selected_verified.json"
    if not path.exists() or not (directory / "selected_terrain_sse.npz").exists():
        return None
    verified = load(path)
    if (verified.get("status") != "verified" or verified.get("official_VAL_used")
            or verified["config_sha256"] != sha(directory / "config.json")
            or verified["export_identity"]["model_sha256"] != exported["model_sha256"]
            or verified["export_identity"]["export_record_sha256"] != sha(directory / "selected_export.json")):
        raise ValueError("Existing independent verification identity differs; inspect rather than rerun")
    for panel in PANELS:
        file = directory / f"selected_{panel}_prediction.npy"
        recorded = load(file.with_suffix(".json"))
        completed = exported["panels"][panel]
        if (completed["file"] != file.name
                or {k: v for k, v in completed.items() if k != "file"} != recorded
                or file.stat().st_size != recorded["file_size_bytes"]
                or verified["panels"][panel]["prediction_sha256"] != recorded["prediction_sha256"]
                or verified["panels"][panel]["prediction_sidecar_sha256"] != sha(file.with_suffix(".json"))):
            raise ValueError("Saved prediction/sidecar/verification mismatch")
        if hash_predictions and sha(file) != recorded["prediction_sha256"]:
            raise ValueError("Saved prediction bytes changed")
        for old_name, name in zip(TRAIN_GROUPS, GROUPS):
            expected = context["selected"]["evaluation"][panel]["groups"][old_name]
            actual = verified["panels"][panel]["groups"][name]
            if (expected["count"] != actual["valid_pixels"]
                    or not np.allclose(expected["RMSE"], actual["RMSE_dB"], rtol=0, atol=1e-9)
                    or not np.allclose(expected["SSE"], actual["SSE_dB2"], rtol=1e-12, atol=1e-6)):
                raise ValueError("Independent selected score and training evaluation differ")
    return verified


def verify(arms):
    plan = protocol()
    contexts = [arm_context(arm, plan) for arm in arms]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for context in contexts:
        existing = evidence(context, hash_predictions=True)
        if existing is not None:
            print(json.dumps({"arm": context["arm"], "independent_verification": "existing_valid_evidence_reused",
                              "network_inference": False}), flush=True)
            continue
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "src.c145_verify",
                        "--run", str(context["directory"].relative_to(ROOT)),
                        "--tag", "selected"], cwd=ROOT, env=env, check=True)
        if evidence(context) is None:
            raise RuntimeError("Independent verifier did not create complete evidence")


def statistics(path, panels, verified):
    with np.load(path, allow_pickle=False) as source:
        if source["group_names"].tolist() != list(GROUPS):
            raise ValueError("Statistic region names differ")
        result = {}
        for panel in PANELS:
            fields = {}
            for suffix in ("map_ids", "map_group_sse", "map_group_baseline_sse",
                           "map_group_count", "terrain_names", "map_terrain_index",
                           "terrain_sse", "terrain_C82_sse", "terrain_count"):
                fields[suffix] = source[f"{panel}_{suffix}"].copy()
            if fields["map_ids"].tolist() != panels[panel]:
                raise ValueError("Statistic map order differs")
            n = len(panels[panel])
            for name in ("map_group_sse", "map_group_baseline_sse", "map_group_count"):
                if fields[name].shape != (len(GROUPS), n, 2):
                    raise ValueError("Unexpected map statistic layout")
            for group, name in enumerate(GROUPS):
                original = verified["panels"][panel]["groups"][name]
                if not np.array_equal(fields["map_group_count"][group].sum(0), original["valid_pixels"]):
                    raise ValueError("Cached group support differs from independent scoring")
                if not np.allclose(fields["map_group_sse"][group].sum(0), original["SSE_dB2"],
                                   rtol=1e-12, atol=1e-6):
                    raise ValueError("Cached group SSE differs from independent scoring")
                if not np.allclose(fields["map_group_baseline_sse"][group].sum(0), original["C82_SSE_dB2"],
                                   rtol=1e-12, atol=1e-6):
                    raise ValueError("Cached C82 SSE differs")
            count = np.zeros_like(fields["terrain_count"])
            sse = np.zeros_like(fields["terrain_sse"])
            np.add.at(count, fields["map_terrain_index"], fields["map_group_count"][0])
            np.add.at(sse, fields["map_terrain_index"], fields["map_group_sse"][0])
            baseline = np.zeros_like(fields["terrain_C82_sse"])
            np.add.at(baseline, fields["map_terrain_index"], fields["map_group_baseline_sse"][0])
            if (not np.array_equal(count, fields["terrain_count"]) or not np.allclose(
                    sse, fields["terrain_sse"], rtol=1e-12, atol=1e-6)
                    or not np.allclose(baseline, fields["terrain_C82_sse"], rtol=1e-12, atol=1e-6)):
                raise ValueError("Terrain sufficient statistics do not reconstruct map totals")
            result[panel] = fields
    return result


def parent_evidence():
    value = load(PARENT / "selected_verified.json")
    expected = "2f13de5e344462264b9184d20ab920d89d645899c2820f0799e334947cb55cd7"
    monitor = value["panels"]["monitor"]["groups"]["all"]
    if (value.get("status") != "verified" or value.get("official_VAL_used")
            or value["export_identity"]["model_sha256"] != expected
            or value["frequency_hz"] != list(FREQUENCIES)
            or monitor["valid_pixels"] != [40864973, 37837148]
            or not np.allclose(monitor["RMSE_dB"], [2.1412210108233403, 2.4082364795626496],
                               rtol=0, atol=1e-10)
            or not np.allclose(monitor["C82_RMSE_dB"], [2.1463905073492664, 2.41801448492553],
                               rtol=0, atol=1e-10)):
        raise ValueError("Frozen C145 selected12/C82 references differ")
    return value


def group_metrics(sse, count, total_sse=None):
    values = {"SSE_dB2": np.asarray(sse, np.float64),
              "valid_pixels": np.asarray(count, np.int64),
              "RMSE_dB": np.sqrt(ratio(sse, count))}
    if total_sse is not None:
        values["share_of_total_SSE_percent"] = 100 * ratio(sse, total_sse)
    return clean(values)


def align(current, parent):
    for panel in PANELS:
        for key in ("map_ids", "map_terrain_index", "terrain_names", "map_group_count", "terrain_count"):
            if not np.array_equal(current[panel][key], parent[panel][key]):
                raise ValueError(f"Paired panel/order/support differs: {panel}/{key}")
        if not np.allclose(current[panel]["map_group_baseline_sse"],
                           parent[panel]["map_group_baseline_sse"], rtol=1e-12, atol=1e-6):
            raise ValueError("Frozen per-map C82 reference differs")


def posthoc_regions(plan, contexts, verified):
    """Score new saved predictions in fixed C149 regions; reuse old reference SSE."""
    old = load(C149 / "summary.json")
    old_protocol = load(C149 / "protocol.json")
    ids = plan["panels"]["monitor"]
    if (old.get("status") != "complete" or old_protocol["ordered_panel_ids"] != ids
            or old["profile_cache_sha256"] != sha(C149 / "profiles.npy")):
        raise ValueError("Fixed C149 profile identity differs")
    parent = parent_evidence()
    if not np.allclose(old["all"]["SSE_dB2"], parent["panels"]["monitor"]["groups"]["all"]["SSE_dB2"],
                       rtol=1e-12, atol=1e-6):
        raise ValueError("C149 attribution refers to another selected12 prediction")
    profiles = np.load(C149 / "profiles.npy", mmap_mode="r", allow_pickle=False)
    if profiles.shape != (628, 5, 256, 256) or profiles.dtype != np.float32:
        raise ValueError("C149 profile layout differs")
    names = ("Clear", "Grazing", "Shallow_Shadow", "Deep_Shadow", "undefined_horizontal_distance")
    sums = {arm: {name: [np.zeros(2), np.zeros(2, np.int64)] for name in names} for arm in ARMS}
    predictions = {arm: np.load(contexts[arm]["directory"] / "selected_monitor_prediction.npy",
                               mmap_mode="r", allow_pickle=False) for arm in ARMS}
    labels = [np.load(ROOT.parent / f"data/train_pl{band}.npy", mmap_mode="r") for band in ("415", "58")]
    validity = [np.load(ROOT.parent / f"data/train_mask{band}.npy", mmap_mode="r") for band in ("415", "58")]
    wavelength = 299792458.0 / np.asarray(FREQUENCIES, np.float64)
    for row, map_id in enumerate(ids):
        target = np.stack([x[map_id] for x in labels]).astype(np.float64)
        valid = np.stack([np.unpackbits(x[map_id], bitorder="big").reshape(256, 256)
                          for x in validity]).astype(bool)
        p = np.asarray(profiles[row])
        nu = p[1][None] / np.sqrt(wavelength[:, None, None])
        defined = (p[4].astype(np.int64) & 1) == 0
        conditions = (defined & (nu < -.7), defined & (nu >= -.7) & (nu < 0),
                      defined & (nu >= 0) & (nu < 1.5), defined & (nu >= 1.5),
                      np.broadcast_to(~defined, valid.shape))
        for arm in ARMS:
            error = np.asarray(predictions[arm][row], np.float64) - target
            for name, condition in zip(names, conditions):
                mask = valid & condition
                sums[arm][name][0] += np.where(mask, error*error, 0).sum((1, 2), dtype=np.float64)
                sums[arm][name][1] += mask.sum((1, 2), dtype=np.int64)
    output = {}
    for arm in ARMS:
        overall = verified[arm]["panels"]["monitor"]["groups"]["all"]
        assert np.array_equal(sum(x[1] for x in sums[arm].values()), overall["valid_pixels"])
        assert np.allclose(sum(x[0] for x in sums[arm].values()), overall["SSE_dB2"], rtol=1e-12, atol=1e-6)
        values = {}
        for name in names:
            sse, count = sums[arm][name]
            ref = old["groups"]["Fresnel_" + name]
            if count.tolist() != ref["valid_pixels"]:
                raise ValueError("C149 fixed region support differs")
            values[name] = {"candidate": group_metrics(sse, count, overall["SSE_dB2"]),
                            "C145_selected12": group_metrics(ref["SSE_dB2"], count, old["all"]["SSE_dB2"]),
                            "C82": group_metrics(ref["C82_SSE_dB2"], count, old["all"]["C82_SSE_dB2"]),
                            "candidate_minus_selected12_SSE_dB2": (sse-np.asarray(ref["SSE_dB2"])).tolist(),
                            "candidate_minus_C82_SSE_dB2": (sse-np.asarray(ref["C82_SSE_dB2"])).tolist()}
        output[arm] = values
    return {"posthoc_diagnostic": True, "used_for_checkpoint_or_arm_selection": False,
            "scope": "Same registered-point C149 masks; Float32 profile thresholds; original validity only for scoring",
            "profile_cache_sha256": old["profile_cache_sha256"],
            "C149_summary_sha256": sha(C149 / "summary.json"), "arms": output,
            "limit": "Bucket associations do not prove diffraction or solver-floor errors caused the target residuals."}


def markdown(summary):
    lines = ["# C150 物理残差目标 A/B 对照", "",
             "同一网络、输入、初始化、数据顺序及12轮预算；唯一差异是输出先验。",
             "A：FSPL + Net；B：FSPL + L_Deygout + Net。此处为重复使用的开发 monitor，非官方 VAL 或盲测。", "",
             "差值统一为 candidate RMSE − reference RMSE，负值表示改善。", "",
             "| 版本 | 选中 Epoch | FIT RMSE 415 / 5.775 GHz (dB) | monitor RMSE 415 / 5.775 GHz (dB) | 双参照双频 CI 门槛 |",
             "|---|---:|---:|---:|---|"]
    pair = lambda x: " / ".join("未定义" if v is None else f"{v:.6f}" for v in x)
    for arm in ARMS:
        item = summary["arms"][arm]
        lines.append(f"| {arm} | {item['selected_epoch']} | {pair(item['selected']['fit']['all']['RMSE_dB'])} | "
                     f"{pair(item['selected']['monitor']['all']['RMSE_dB'])} | {'通过' if item['gate_passed'] else '未通过'} |")
    lines += ["", "## 地形配对 95% CI", "",
              "| 候选 | 参照 | ΔRMSE 415 / 5.775 GHz (dB) | 415 MHz 95% CI | 5.775 GHz 95% CI |",
              "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        for reference, comparison in summary["arms"][arm]["paired_comparisons"].items():
            ci = comparison["delta_RMSE_dB_95CI"]
            lines.append(f"| {arm} | {reference} | {pair(comparison['point_delta_RMSE_dB'])} | "
                         f"[{ci[0][0]:.6f}, {ci[0][1]:.6f}] | [{ci[1][0]:.6f}, {ci[1][1]:.6f}] |")
    lines += ["", "## 固定误差区域", "",
              "| 版本 | 面板 | 区域 | RMSE 415 / 5.775 GHz (dB) |",
              "|---|---|---|---:|"]
    for arm in ARMS:
        for panel in PANELS:
            for group in ("all", "tail", "C82_good"):
                values = summary["arms"][arm]["selected"][panel][group]
                lines.append(f"| {arm} | {panel} | {group} | {pair(values['RMSE_dB'])} |")
    lines += ["", "## C149 固定 Fresnel 分组", "",
              "以下只用于事后解释，不参与选中 checkpoint。贡献为该组 SSE / 该候选全部 monitor SSE。", "",
              "| 版本 | 区域 | RMSE 415 / 5.775 GHz (dB) | 总MSE贡献 415 / 5.775 GHz (%) |",
              "|---|---|---:|---:|"]
    for arm in ARMS:
        for name in ("Grazing", "Shallow_Shadow"):
            value = summary["posthoc_C149_regions"]["arms"][arm][name]["candidate"]
            lines.append(f"| {arm} | {name} | {pair(value['RMSE_dB'])} | {pair(value['share_of_total_SSE_percent'])} |")
    lines += ["", "## 判断", "",
              "允许扩大预算：" + ("是；至少一项满足预注册门槛。" if summary["budget_scale_allowed"]
                                 else "否；本次固定预算结果未通过双参照双频门槛。"),
              "C151a / C151b 是另行规定的新机制固定预算对照；本报告不自动延长 C150 或扫描参数。",
              "selected 由保存预测独立 CPU FP64 核验；last 只有在其与 selected 为同一预测器时沿用该独立证据，其他 last 数值来自原训练评估。",
              "完整来源、逐频支持数、初始/末轮与 A/B 配对比较见 comparison_summary.json。", ""]
    return "\n".join(lines)


def log_report(summary, previous, enabled=True):
    core = json.dumps(clean(summary), sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(core.encode()).hexdigest()
    if (previous and previous.get("tracking", {}).get("analysis_payload_sha256") == identity
            and previous["tracking"].get("status") == "complete"):
        return previous["tracking"]
    if not enabled:
        return {"status": "not_requested", "reason": "Training records already local; explicit report --no-swanlab"}
    from src import cc_tracking
    run = cc_tracking.init(name="C150 verified paired comparison", campaign="C150",
        config={"run_kind": "analysis.paired_comparison", "analysis_payload_sha256": identity,
                "summary": "audit/c150_diffraction_training/comparison_summary.json",
                "protocol_sha256": summary["protocol_sha256"],
                "training_sources": {arm: summary["arms"][arm]["tracking"] for arm in ARMS},
                "_cc_dashboard": {"kind": "analysis", "scope": "paired saved-prediction C150 comparison; no training"}},
        description="Independent CPU scoring and paired terrain bootstrap; candidate minus references; local only.",
        tags=["analysis", "paired-terrain", "no-new-training"])
    run_identity = {"run_id": run.id, "dir": str(run.dir)}
    values = {}
    for arm in ARMS:
        values[f"gate/{arm}_passed"] = int(summary["arms"][arm]["gate_passed"])
        for reference, comparison in summary["arms"][arm]["paired_comparisons"].items():
            for k, band in enumerate(BANDS):
                prefix = f"paired/{arm}/{reference}/{band}"
                values[prefix + "/delta_RMSE_dB"] = comparison["point_delta_RMSE_dB"][k]
                values[prefix + "/CI95_lower_dB"] = comparison["delta_RMSE_dB_95CI"][k][0]
                values[prefix + "/CI95_upper_dB"] = comparison["delta_RMSE_dB_95CI"][k][1]
    try:
        cc_tracking.log(values, step=0)
    finally:
        cc_tracking.finish()
    return {"status": "complete", "analysis_payload_sha256": identity,
            **run_identity, "mode": "local"}


def summary(log_swanlab=True):
    plan = protocol()
    contexts = {arm: arm_context(arm, plan) for arm in ARMS}
    verified = {}
    current = {}
    parent_verified = parent_evidence()
    parent = statistics(PARENT / "selected_terrain_sse.npz", plan["panels"], parent_verified)
    for arm in ARMS:
        verified[arm] = evidence(contexts[arm], hash_predictions=True)
        if verified[arm] is None:
            raise ValueError("Run c150_report verify/all before summary")
        current[arm] = statistics(contexts[arm]["directory"] / "selected_terrain_sse.npz",
                                  plan["panels"], verified[arm])
        align(current[arm], parent)
    output = {"status": "verified", "campaign": "C150",
              "protocol_sha256": sha(AUDIT / "protocol.json"),
              "report_source_sha256": sha(Path(__file__)),
              "frequency_hz": list(FREQUENCIES), "scope": plan["scope"],
              "gate_definition": plan["gate"], "official_VAL_used": False,
              "training_or_network_inference_repeated": False,
              "arms": {}, "references": {},
              "limitations": ["One fixed paired seed/budget, no sweep or extension.",
                  "C82 beta-fit80 and model-selection48 terrains are all development data.",
                  "CIs condition on repeated monitor use and checkpoint selection; no fresh-blind claim.",
                  "Initial analytic-prior checkpoint may win and is reported honestly."]}
    output["references"]["C145_selected12"] = {
        "verified_source": str((PARENT / "selected_verified.json").relative_to(ROOT)),
        "verified_sha256": sha(PARENT / "selected_verified.json"),
        "statistic_sha256": sha(PARENT / "selected_terrain_sse.npz"),
        "model_sha256": parent_verified["export_identity"]["model_sha256"]}
    for arm in ARMS:
        context = contexts[arm]
        st = current[arm]["monitor"]
        comparisons = {
            "C82": paired_delta(st["terrain_sse"], parent["monitor"]["terrain_C82_sse"], st["terrain_count"]),
            "C145_selected12": paired_delta(st["terrain_sse"], parent["monitor"]["terrain_sse"], st["terrain_count"])}
        groups = {}
        for panel in PANELS:
            groups[panel] = {}
            for old_name, name in zip(TRAIN_GROUPS, GROUPS):
                raw = verified[arm]["panels"][panel]["groups"][name]
                groups[panel][old_name] = {key: raw[key] for key in
                    ("RMSE_dB", "SSE_dB2", "valid_pixels", "C82_RMSE_dB", "C82_SSE_dB2")}
        last = context["history"][-1]
        item = {"prior": plan["arms"][arm], "selected_epoch": context["selected"]["epoch"],
                "selected_initial_analytic_checkpoint": context["selected"]["epoch"] == 0,
                "predictor_sha256": context["export"]["model_sha256"],
                "net_state_sha256": context["export"]["net_state_sha256"],
                "selected": groups, "paired_comparisons": comparisons,
                "gate_passed": all(x["both_band_upper_CI_strictly_below_zero"] for x in comparisons.values()),
                "last_epoch": 12, "last_original_training_evaluation": last["evaluation"],
                "last_independently_verified": last["predictor_sha256"] == context["export"]["model_sha256"],
                "initial_original_training_evaluation": context["history"][0]["evaluation"],
                "selected_vs_last_monitor_delta_RMSE_dB":
                    (np.asarray(last["evaluation"]["monitor"]["rmse"])
                     - np.asarray(context["selected"]["evaluation"]["monitor"]["rmse"])).tolist(),
                "tracking": load(context["directory"] / "tracking.json"),
                "evidence": {"verified_json": str((context["directory"] / "selected_verified.json").relative_to(ROOT)),
                    "verified_json_sha256": sha(context["directory"] / "selected_verified.json"),
                    "statistic_sha256": sha(context["directory"] / "selected_terrain_sse.npz"),
                    "config_sha256": sha(context["directory"] / "config.json"),
                    "history_sha256": sha(context["directory"] / "history.json"),
                    "selected_export_sha256": sha(context["directory"] / "selected_export.json")}}
        output["arms"][arm] = item
    output["A_B_pair"] = {"difference": "B_RMSE_minus_A_RMSE_dB",
        "selected_checkpoints": {arm: output["arms"][arm]["selected_epoch"] for arm in ARMS},
        "paired": paired_delta(current["B"]["monitor"]["terrain_sse"],
                               current["A"]["monitor"]["terrain_sse"],
                               current["A"]["monitor"]["terrain_count"]),
        "interpretation_limit": "Compares the fixed prior interventions with equal training/selection budgets; it does not identify the cause of original target errors."}
    output["posthoc_C149_regions"] = posthoc_regions(plan, contexts, verified)
    output["budget_scale_allowed"] = any(output["arms"][arm]["gate_passed"] for arm in ARMS)
    output["passing_arms"] = [arm for arm in ARMS if output["arms"][arm]["gate_passed"]]
    output["lowest_development_joint_mean_arm"] = min(
        ARMS, key=lambda arm: np.mean(output["arms"][arm]["selected"]["monitor"]["all"]["RMSE_dB"]))
    previous = load(AUDIT / "comparison_summary.json") if (AUDIT / "comparison_summary.json").exists() else None
    output = clean(output)
    output["tracking"] = log_report(output, previous, log_swanlab)
    atomic(AUDIT / "comparison_summary.json", output)
    (AUDIT / "comparison_summary.md").write_text(markdown(output))
    print(json.dumps({"status": output["status"],
                      "selected": {arm: {"epoch": output["arms"][arm]["selected_epoch"],
                          "RMSE_dB": output["arms"][arm]["selected"]["monitor"]["all"]["RMSE_dB"],
                          "gate": output["arms"][arm]["gate_passed"]} for arm in ARMS},
                      "budget_scale_allowed": output["budget_scale_allowed"],
                      "summary": "audit/c150_diffraction_training/comparison_summary.json"}, ensure_ascii=False),
          flush=True)


def preflight():
    verified = parent_evidence()
    panels = load(ROOT / "audit/c145_capacity/protocol.json")["panels"]
    st = statistics(PARENT / "selected_terrain_sse.npz", panels, verified)
    zero = paired_delta(st["monitor"]["terrain_sse"], st["monitor"]["terrain_sse"],
                        st["monitor"]["terrain_count"])
    assert zero["point_delta_RMSE_dB"] == [0., 0.]
    assert zero["delta_RMSE_dB_95CI"] == [[0., 0.], [0., 0.]]
    assert not zero["both_band_upper_CI_strictly_below_zero"]
    count = np.full((8, 2), 100, np.int64)
    old = np.full((8, 2), 4., np.float64) * count
    better = np.full((8, 2), 1., np.float64) * count
    result = paired_delta(better, old, count)
    assert np.allclose(result["point_delta_RMSE_dB"], [-1, -1])
    assert np.allclose(result["delta_RMSE_dB_95CI"], [[-1,-1],[-1,-1]])
    assert result["both_band_upper_CI_strictly_below_zero"]
    mixed = better.copy();mixed[:, 1] = old[:, 1]
    assert not paired_delta(mixed, old, count)["both_band_upper_CI_strictly_below_zero"]
    assert "torch" not in sys.modules
    print(json.dumps({"status": "CPU_report_preflight_passed", "reference_terrains": len(st["monitor"]["terrain_names"]),
                      "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
                      "zero_difference_gate": False, "negative_difference_fixture_gate": True,
                      "single_band_fixture_gate": False, "Torch_imported": False,
                      "report_source_sha256": sha(Path(__file__))}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("preflight", "verify", "summary", "all"),
                        default="summary")
    parser.add_argument("--arm", choices=("A", "B", "both"), default="both")
    parser.add_argument("--no-swanlab", action="store_true",
                        help="Explicitly omit derived-comparison run; original training runs remain local")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
        return
    arms = ARMS if args.arm == "both" else (args.arm,)
    if args.command in ("verify", "all"):
        verify(arms)
    if args.command in ("summary", "all"):
        if args.arm != "both":
            parser.error("Aggregate summary requires both completed arms")
        summary(not args.no_swanlab)


if __name__ == "__main__":
    main()
