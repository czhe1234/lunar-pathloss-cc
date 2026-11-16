"""Read-only research overview for the local SwanLab service. No model imports."""
from __future__ import annotations
import json
import hashlib
import math
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "tracking/swanlab"
SOURCES = {
    "c145_summary": "audit/c145_capacity/comparison_summary.json",
    "c145_history": "runs/c145_capacity_native/history.json",
    "c145_initial": "runs/c145_capacity_native/initial.json",
    "c145_phases": "runs/c145_capacity_native/phases.json",
    "c146_summary": "audit/c146_solver_ablation/comparison_summary.json",
    "c146_history": "runs/c146_solver_ablation/history.json",
    "c146_complete": "runs/c146_solver_ablation/training_complete.json",
    "c147_summary": "audit/c147_sionna_reflections/comparison_direct_covered.json",
}
GROUPS = ("all", "tail", "C82_good")
G145 = {"all": "all", "tail": "C82_abs_error_ge_5dB", "C82_good": "C82_abs_error_le_1dB"}
_LOCK = threading.Lock()
_CACHE = (0.0, None)

def read(key):
    return json.loads((ROOT / SOURCES[key]).read_text())

def gains(base, value):
    if not isinstance(base, (list, tuple)) or not isinstance(value, (list, tuple)):
        return None
    if len(base) != 2 or len(value) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*base, *value]):
        return None
    return [b - v for b, v in zip(base, value)]

def result_row(label, epoch, fit, evaluation, baseline, tail, tail_baseline,
               cost_seconds=None, cost_note="", kind="candidate"):
    gain = gains(baseline, evaluation)
    return {
        "label": label, "epoch": epoch, "fit": fit, "evaluation": evaluation,
        "C82": baseline, "gain": gain, "tail": tail, "tail_gain": gains(tail_baseline, tail),
        "beats_both": all(x > 0 for x in gain) if gain is not None else None,
        "cost_seconds": cost_seconds, "cost_note": cost_note, "kind": kind,
    }

def verified145(source, tag, panel, group):
    return source["verified_results"][tag]["panels"][panel]["groups"][G145[group]]

def baseline_row(panel):
    return result_row("C82 参照", None, None, panel["C82"][0], panel["C82"][0],
                      panel["C82"][1], panel["C82"][1], kind="baseline")

def attach_c148_evidence(profile, cfg, record):
    """Attach only the completed C148 comparison; preserve native progress and curves."""
    summary_path = ROOT / "audit/c148_generalization/comparison_summary.json"
    if cfg.get("campaign") != "C148" or record["status"] != 1 or not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text())
        config_path = ROOT / "runs/c148_parameter_anchor/config.json"
        config = json.loads(config_path.read_text())
        result = json.loads((config_path.parent / "training_result.json").read_text())
        tracking = json.loads((config_path.parent / "tracking.json").read_text())
        protocol_path = ROOT / "audit/c148_generalization/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        original = json.loads((ROOT / "runs/c145_capacity_native/config.json").read_text())
        base_path = ROOT / protocol["base_protocol"]
        base = json.loads(base_path.read_text())
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        protocol_sha = sha(protocol_path)
        if (summary.get("status") != "verified"
                or result.get("status") != "complete" or not result.get("fixed_budget_complete")
                or summary.get("official_VAL_used") or config.get("official_VAL_used")
                or summary.get("protocol_sha256") != protocol_sha
                or config.get("protocol_sha256") != protocol_sha
                or result.get("protocol_sha256") != protocol_sha
                or cfg.get("protocol_sha256") != protocol_sha
                or summary.get("config_sha256") != sha(config_path)
                or protocol["base_protocol_sha256"] != sha(base_path)
                or config["panels"] != original["panels"] or config["panels"] != base["panels"]
                or cfg.get("panels") != config["panels"]
                or Path(tracking["dir"]).name != record["run_id"]
                or summary["independent_prediction_evidence"].get("status") != "verified"):
            return []
        # A complete summary is tied to the actual selected/last epochs and original masks.
        final_epoch = protocol["parent_absolute_epoch"] + protocol["additional_epochs"]
        selected_epoch = summary["selected_absolute_epoch"]
        if (result["absolute_epoch"] != final_epoch
                or result["selected_absolute_epoch"] != selected_epoch
                or summary["scope"] != protocol["evaluation_scope"]):
            return []
        checkpoints, rows = {}, []
        for tag, title, epoch in (("selected", "选中权重", selected_epoch),
                                  ("last", "最后权重", final_epoch)):
            metrics = summary[tag]
            for panel in ("fit", "monitor"):
                for group in ("all", "tail", "C82_good"):
                    current = metrics[panel][group]
                    parent = summary["parent_selected12"][panel][group]
                    if current["valid_pixels"] != parent["valid_pixels"]:
                        return []
                    if not all(math.isclose(x, y, rel_tol=0, abs_tol=1e-12)
                               for x, y in zip(current["C82_RMSE_dB"], parent["C82_RMSE_dB"])):
                        return []
            monitor, tail = metrics["monitor"]["all"], metrics["monitor"]["tail"]
            row = result_row("C148 · " + title, epoch, metrics["fit"]["all"]["RMSE_dB"],
                             monitor["RMSE_dB"], monitor["C82_RMSE_dB"], tail["RMSE_dB"],
                             tail["C82_RMSE_dB"], profile["elapsed_seconds"],
                             "本次续训累计，含评测；两权重共享该耗时")
            row.update(profile=profile["id"], checkpoint=tag)
            regions = []
            for panel, panel_title in (("fit", "训练集"), ("monitor", "留出开发地形")):
                for group, title_region in (("all", "全部"), ("tail", "固定大误差区"),
                                             ("C82_good", "原本较准区域")):
                    value = metrics[panel][group]
                    regions.append({"label": panel_title + " · " + title_region,
                                    "candidate": value["RMSE_dB"], "baseline": value["C82_RMSE_dB"],
                                    "gain": gains(value["C82_RMSE_dB"], value["RMSE_dB"]),
                                    "count": value["valid_pixels"]})
            checkpoints[tag] = {"row": row, "regions": regions}
            rows.append(row)
        gain = summary["selected_vs_selected12"]["monitor"]["all"]["gain_reference_minus_C148_dB"]
        uncertainty = summary.get("paired_terrain_gain_over_selected12", {})
        interval = (uncertainty.get("gain_selected12_minus_C148_dB_95CI")
                    if uncertainty.get("status") == "verified" else None)
        crosses_zero = any(low <= 0 <= high for low, high in interval) if interval else None
        improved = all(value > 0 for value in gain)
        tiny = improved and max(abs(value) for value in gain) < .01
        if summary["selected_remains_parent"]:
            decision = "选中权重仍为 selected12，未取得新模型改善"
        elif tiny:
            decision = "停止追加本配置训练：改进很小"
        elif improved:
            decision = "开发集有数值改善，仍需结合不确定性判断"
        else:
            decision = "未在同一权重上取得双频改善"
        selected_rmse = summary["selected"]["monitor"]["all"]["RMSE_dB"]
        last_rmse = summary["last"]["monitor"]["all"]["RMSE_dB"]
        worse = [last_rmse[i] > selected_rmse[i] for i in range(2)]
        late_note = ("末轮相对选中权重双频恶化，后期仍有退化。" if all(worse)
                     else "末轮相对选中权重至少一频恶化，后期仍有退化。" if any(worse)
                     else "末轮两频误差均未高于选中权重。")
        ci_note = ("至少一频的 95% CI 跨 0，尚不能确认稳定改善。" if crosses_zero
                   else "两频 95% CI 均不跨 0；这是反复使用的开发集诊断。" if interval
                   else "配对地形 95% CI 尚未核验。")
        evidence = ["selected：保存预测的独立 CPU 核验；" +
                    summary["independent_prediction_evidence"]["source"],
                    "last：原始训练评估，GPU FP64 SSE/count；未做保存预测的独立核验",
                    "协议 hash 与有序面板 ID 一致；C82 beta 拟合80、选择48均为开发数据"]
        if interval:
            evidence.append("selected 相对 selected12 的双频 95% CI (dB)："
                            + "；".join(f"[{low:.8f}, {high:.8f}]" for low, high in interval))
        profile.update(checkpoints=checkpoints, best_epoch=selected_epoch,
                       method_note="C145 selected12 全状态续训，仅增加参数锚定 L2-SP。",
                       decision=decision,
                       reason=f"相对 selected12 的双频改善为 {gain[0]:.8f} / {gain[1]:.8f} dB。"
                              + ci_note + late_note + "两组均为开发数据，不能视为盲测。",
                       evidence=evidence, sources=["c148_summary"])
        SOURCES["c148_summary"] = str(summary_path.relative_to(ROOT))
        return rows
    except (OSError, KeyError, TypeError, ValueError):
        # Incomplete/unmatched evidence retains the original pending native profile.
        return []


def native_history_paths():
    """Associate only current run metadata; never search historical archives."""
    found = {}
    for path in [* (ROOT / "runs").glob("*/tracking.json"),
                 * (ROOT / "runs").glob("*/*/tracking.json")]:
        try:
            tracking = json.loads(path.read_text())
            found[Path(tracking["dir"]).name] = path.parent
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return found


def attach_native_history(profile, cfg, record, directory):
    """Read committed evaluation/selection metadata without selecting or scoring."""
    if directory is None or cfg.get("campaign") == "C148":
        return []
    try:
        config = json.loads((directory / "config.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        result_path, export_path = directory / "result.json", directory / "selected_export.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        exported = json.loads(export_path.read_text()) if export_path.exists() else {}
        if (not history or config.get("campaign") != cfg.get("campaign")
                or config.get("arm") != cfg.get("arm")
                or config.get("protocol_sha256") != cfg.get("protocol_sha256")
                or config.get("panels") != cfg.get("panels")):
            return []
        if result and (result.get("protocol_sha256") != config.get("protocol_sha256")
                       or result.get("arm") != config.get("arm")):
            return []
        last = history[-1]
        selected_epoch = result.get("selected_epoch", exported.get("training_state", {}).get("epoch"))
        selected = next((point for point in history if point["epoch"] == selected_epoch), None)
        if (selected is not None and
                (not (directory / exported.get("checkpoint", "")).is_file()
                 or exported.get("training_state", {}).get("epoch") != selected_epoch
                 or exported.get("model_sha256") != selected.get("predictor_sha256"))):
            selected = None
        complete = result.get("status") == "complete"
        if complete and (last["epoch"] != result["epochs"]
                         or last["evaluation"]["monitor"]["rmse"] != result["last_monitor_RMSE_dB"]
                         or selected is None
                         or selected["evaluation"]["monitor"]["rmse"] != result["selected_monitor_RMSE_dB"]):
            return []
        elapsed = result.get("elapsed_seconds", last.get("elapsed_seconds", profile["elapsed_seconds"]))
        checkpoints, rows = {}, []
        for tag, title, point in (("selected", "选中权重", selected),
                                  ("last", "最后权重" if complete else "最新记录", last)):
            if point is None:
                continue
            evaluation = point["evaluation"]
            monitor = evaluation["monitor"]
            group = monitor.get("groups", {})
            all_group, tail = group.get("all", {}), group.get("tail", {})
            baseline = all_group.get("C82_RMSE", profile["baseline"].get("monitor"))
            row = result_row(record["name"] + " · " + title, point["epoch"],
                             evaluation["fit"]["rmse"], monitor["rmse"], baseline,
                             tail.get("RMSE"), tail.get("C82_RMSE"), elapsed,
                             "整次训练累计，含评测；两版本共享该耗时")
            row.update(profile=profile["id"], checkpoint=tag)
            regions = []
            for panel, panel_title in (("fit", "训练集"), ("monitor", "留出地形")):
                for name, title_region in (("all", "全部"), ("tail", "固定大误差区"),
                                            ("C82_good", "原本较准区域")):
                    value = evaluation[panel].get("groups", {}).get(name)
                    if value is not None:
                        regions.append({"label": panel_title + " · " + title_region,
                            "candidate": value["RMSE"], "baseline": value.get("C82_RMSE"),
                            "gain": gains(value.get("C82_RMSE"), value["RMSE"]), "count": value["count"]})
            checkpoints[tag] = {"row": row, "regions": regions}
            rows.append(row)
        profile.update(checkpoints=checkpoints, best_epoch=selected["epoch"] if selected else None,
                       epoch=last["epoch"], updates=last.get("updates", profile["updates"]),
                       elapsed_seconds=elapsed, elapsed_note="原始 history/result 记录，含评测",
                       series={panel: [[point["epoch"], *point["evaluation"][panel]["rmse"]]
                                       for point in history] for panel in ("fit", "monitor")},
                       method_note=(cfg.get("_cc_dashboard") or {}).get("method_note","同一数据划分；输入与输出先验定义见原实验配置。"),
                       decision="固定预算已完成，不自动延长训练" if complete else "按既定训练预算继续",
                       reason="选中轮次直接读取原实验保存记录，按双频联合规则选出；"
                              "选中与末轮指标均来自原始评估，独立复核状态以实验报告为准。",
                       evidence=["未重算评测或补造中间曲线；区域数据直接读取原 history.json。"])
        if complete:
            profile.update(status="complete", eta_seconds=0,
                           progress_text=f"已完成 · {last['epoch']} / {result['epochs']} 轮")
        else:
            profile["progress_text"] = str(last["epoch"]) + " 轮（最近一次完整评估）"
            progress_path = directory / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                if (progress.get("stage") == "training" and progress.get("arm") == config.get("arm")
                        and progress.get("updates", -1) >= profile["updates"]):
                    for key in ("epoch", "updates", "elapsed_seconds", "eta_seconds"):
                        if progress.get(key) is not None:
                            profile[key] = progress[key]
                    profile["progress_text"] = (f"第 {progress['epoch']} / {progress['epochs']} 轮"
                                                f" · {progress['batch']} / {progress['batches']} 批")
                    profile["elapsed_note"] = "原始 progress.json 实时进度；曲线仍仅显示完整评估"
        for name, path in (("history", directory / "history.json"), ("result", result_path),
                           ("progress", directory / "progress.json")):
            if path.exists():
                key = profile["id"] + "_" + name
                SOURCES[key] = str(path.relative_to(ROOT))
                profile["sources"].append(key)
        # Only the original common ordered panel may enter the shared C82 table.
        common = json.loads((ROOT / "audit/c145_capacity/protocol.json").read_text())["panels"]
        return rows if config.get("panels") == common else []
    except (OSError, KeyError, TypeError, ValueError):
        return []


def build():
    # Optional final evidence is exposed only while the current validation passes.
    SOURCES.pop("c148_summary", None)
    for key in list(SOURCES):
        if key.startswith("native_"):
            SOURCES.pop(key)
    a, h145, initial, phases = read("c145_summary"), read("c145_history"), read("c145_initial"), read("c145_phases")
    b, h146, done146, physics = read("c146_summary"), read("c146_history"), read("c146_complete"), read("c147_summary")
    assert a["status"] == "complete_identity_checked" and b["status"] == "verified"
    # Only these matched panels may share a comparison table.
    for panel in ("fit", "monitor"):
        assert initial[panel]["groups"]["all"]["count"] == b["panels"][panel]["count"][0]
        assert all(abs(x-y) < 1e-12 for x,y in zip(initial[panel]["C82_rmse"], b["panels"][panel]["C82"][0]))
    native_db = sqlite3.connect("file:" + str(STATE / "logs/runs.swanlab") + "?mode=ro", uri=True)
    native_db.row_factory = sqlite3.Row
    native = [dict(x) for x in native_db.execute("select id,name,run_id,status,create_time,finish_time from experiment")]
    native_by_dir = {x["run_id"]: x for x in native}
    manifest = json.loads((STATE / "migration/manifest.json").read_text())
    imported = {v["run_dir"].split("/")[-1]: v for v in manifest["runs"].values()}
    def native_link(prefix):
        record = next(v for v in manifest["runs"].values() if v["name"].startswith(prefix))
        return "/experiment/" + str(native_by_dir[Path(record["run_dir"]).name]["id"]) + "/chart"
    elapsed145 = sum(p.get("elapsed_seconds", 0) for p in phases if not p.get("export_only"))
    elapsed146 = h146[-1]["seconds"]
    profiles = []
    shared_rows = [baseline_row(b["panels"]["monitor"])]
    confirm_rows = [baseline_row(b["panels"]["confirmation"])]
    base_monitor = b["panels"]["monitor"]["C82"][0]
    scope_note = "地形留出开发评测：628 张图 / 128 个地形；相对当前分支 FIT 留出。反复用于开发，C82 校准曾见其中 80 个地形。"
    points145 = [{"epoch": 8, "evaluation": initial}, *h145]
    series145 = {"fit": [], "monitor": [], "fit_tail": [], "monitor_tail": []}
    for point in points145:
        for panel in ("fit", "monitor"):
            ev = point["evaluation"][panel]
            series145[panel].append([point["absolute_epoch"] if "absolute_epoch" in point else point["epoch"], *ev["rmse"]])
            series145[panel + "_tail"].append([point["absolute_epoch"] if "absolute_epoch" in point else point["epoch"], *ev["groups"]["tail"]["RMSE"]])
    checkpoints145 = {}
    for tag, title in (("selected", "C145 · 选中权重"), ("last", "C145 · 最后权重")):
        group = verified145(a, tag, "monitor", "all")
        tail = verified145(a, tag, "monitor", "tail")
        row = result_row(title, a["checkpoint_epochs"][tag]["absolute_epoch"],
                         verified145(a, tag, "fit", "all")["RMSE_dB"], group["RMSE_dB"], group["C82_RMSE_dB"],
                         tail["RMSE_dB"], tail["C82_RMSE_dB"], elapsed145,
                         "续训各阶段累计，含阶段内评测；不含 C123 起点 8 轮")
        row.update(profile="c145", checkpoint=tag)
        shared_rows.append(row)
        regions = []
        for panel, group_name, title_region in (
            ("fit", "all", "训练集 · 全部"), ("fit", "tail", "训练集 · 固定大误差区"),
            ("monitor", "all", "留出地形 · 全部"), ("monitor", "tail", "留出地形 · 固定大误差区"),
            ("monitor", "C82_good", "留出地形 · 原本较准区域")):
            g = verified145(a, tag, panel, group_name)
            regions.append({"label": title_region, "candidate": g["RMSE_dB"], "baseline": g["C82_RMSE_dB"],
                            "gain": gains(g["C82_RMSE_dB"], g["RMSE_dB"]), "count": g["valid_pixels"]})
        checkpoints145[tag] = {"row": row, "regions": regions}
    last145, best145 = checkpoints145["last"]["row"], checkpoints145["selected"]["row"]
    profiles.append({
        "id": "c145", "label": "C145 · 更强模型充分训练", "kind": "training", "status": "complete",
        "method_note": "冻结 C82 + 15.2M 参数残差分支。",
        "native": native_link("C145"), "epoch": a["training"]["absolute_epoch"],
        "progress_text": "已完成 · 新增 756 轮 / 总计 764 轮", "updates": a["training"]["additional_step"],
        "elapsed_seconds": elapsed145, "elapsed_note": "续训阶段累计，含阶段内评测",
        "eta_seconds": 0, "best_epoch": best145["epoch"], "x_title": "总训练轮数",
        "series": series145, "baseline": {p: initial[p]["C82_rmse"] for p in ("fit", "monitor")},
        "checkpoints": checkpoints145, "scope_note": scope_note,
        "decision": "不优先按原配置继续加时",
        "reason": "选中权重在总第 12 轮。继续到 764 轮后，训练集变好，留出地形双频误差变差；已满足原停止条件。",
        "evidence": ["学习率已下降 4 次", "FIT 停滞 16 轮", "最近一次降学习率后已训练 20 轮"],
        "sources": ["c145_summary", "c145_history", "c145_phases"],
    })
    for arm, label in (("with_solver", "C146 · 使用求解器特征"), ("without_solver", "C146 · 不用求解器特征")):
        checkpoints = {}
        for tag, title, model_key in (("selected", "选中权重", arm + "_selected"), ("last", "最后权重", arm + "_final")):
            m = b["panels"]["monitor"]["models"][model_key]
            row = result_row(label + " · " + title, m["epoch"], b["panels"]["fit"]["models"][model_key]["RMSE"]["all"],
                             m["RMSE"]["all"], base_monitor, m["RMSE"]["tail"],
                             b["panels"]["monitor"]["C82"][1], elapsed146, "两臂配对训练合计，不能分别当成独立耗时")
            row.update(profile="c146_" + arm, checkpoint=tag)
            shared_rows.append(row)
            regions = []
            for panel, group_name, region_label in (
                ("fit", "all", "训练集 · 全部"), ("fit", "tail", "训练集 · 固定大误差区"),
                ("monitor", "all", "留出地形 · 全部"), ("monitor", "tail", "留出地形 · 固定大误差区"),
                ("monitor", "C82_good", "留出地形 · 原本较准区域")):
                panel_data = b["panels"][panel]
                idx = panel_data["groups"].index(group_name)
                value = panel_data["models"][model_key]["RMSE"][group_name]
                baseline = panel_data["C82"][idx]
                regions.append({"label": region_label, "candidate": value, "baseline": baseline,
                                "gain": gains(baseline, value), "count": panel_data["count"][idx]})
            checkpoints[tag] = {"row": row, "regions": regions}
            cm = b["panels"]["confirmation"]["models"][model_key]
            cp = b["panels"]["confirmation"]
            cr = result_row(label + " · " + title, cm["epoch"], None, cm["RMSE"]["all"], cp["C82"][0],
                            cm["RMSE"]["tail"], cp["C82"][1], elapsed146, "两臂配对训练合计")
            cr.update(profile="c146_" + arm, checkpoint=tag)
            confirm_rows.append(cr)
        series = {"fit": [], "monitor": []}
        for point in h146:
            for panel in series:
                if panel in point:
                    series[panel].append([point["epoch"], *point[panel][arm]["rmse"]])
        profiles.append({
            "id": "c146_" + arm, "label": label, "kind": "training", "status": "complete",
            "method_note": "15.2M 参数网络；训练与推理均移除 C82，仅改变快速求解器特征。",
            "native": native_link("C146 | " + arm),
            "epoch": done146["epochs"], "progress_text": "已完成 · 128 / 128 轮",
            "updates": done146["updates_per_arm"], "elapsed_seconds": elapsed146,
            "elapsed_note": "两臂配对训练合计", "eta_seconds": 0,
            "best_epoch": checkpoints["selected"]["row"]["epoch"], "x_title": "训练轮数",
            "series": series, "baseline": {"fit": b["panels"]["fit"]["C82"][0], "monitor": base_monitor},
            "checkpoints": checkpoints, "scope_note": scope_note,
            "decision": "当前无延长同配置训练的依据",
            "reason": "固定 128 轮预算已完成。选中与最后权重都未双频超过 C82；延长训练的价值需由新的受控方案支持。",
            "evidence": ["两臂同网络、同划分、同预算", "确认集也未超过 C82"],
            "sources": ["c146_summary", "c146_history", "c146_complete"],
        })
    # Future runs are shown from native records. Missing references stay missing.
    history_paths = native_history_paths()
    for record in native:
        if record["run_id"] in imported:
            continue
        cfg_file = STATE / "logs" / record["run_id"] / "files/config.yaml"
        if not cfg_file.exists():
            continue
        cfg = {k: v.get("value") for k, v in (yaml.safe_load(cfg_file.read_text()) or {}).items()}
        if "_cc_tracking" not in cfg:
            continue
        dc = cfg.get("_cc_dashboard") or {}
        if not isinstance(dc, dict):
            dc = {}
        if (str(cfg.get("run_kind", "")).startswith("analysis.") or dc.get("kind") == "analysis"
                or (history_paths.get(record["run_id"]) and (history_paths[record["run_id"]] / "invalid_attempt.json").exists())):
            continue
        keys = ["progress/epoch", "progress/total_epochs", "progress/updates", "time/elapsed_seconds", "time/eta_seconds",
                "fit/RMSE_415MHz_dB", "fit/RMSE_5p775GHz_dB",
                "monitor/RMSE_415MHz_dB", "monitor/RMSE_5p775GHz_dB"]
        metrics = {}
        for tag in native_db.execute("select name,folder,type from tag where experiment_id=?", (record["id"],)):
            if tag["name"] not in keys or tag["type"] != "line":
                continue
            vals = []
            for path in (STATE / "logs" / record["run_id"] / "logs" / tag["folder"]).glob("*.log"):
                for line in path.read_text().splitlines():
                    try:
                        p = json.loads(line); vals.append([p["index"], p["data"]])
                    except (json.JSONDecodeError, KeyError):
                        continue
            metrics[tag["name"]] = sorted(vals)
        def latest(key):
            return metrics[key][-1][1] if metrics.get(key) else None
        paired = {}
        for panel in ("fit", "monitor"):
            left = dict(metrics.get(panel + "/RMSE_415MHz_dB", []))
            right = dict(metrics.get(panel + "/RMSE_5p775GHz_dB", []))
            paired[panel] = [[x, left[x], right[x]] for x in sorted(left.keys() & right.keys())]
        epoch = latest("progress/epoch")
        total = latest("progress/total_epochs") or dc.get("total_epochs")
        baseline = (dc.get("C82") or {}) if dc.get("scope") and dc.get("baseline_source") else {}
        if not isinstance(baseline, dict):
            baseline = {}
        evaluation = paired["monitor"][-1][1:] if paired["monitor"] else None
        fit = paired["fit"][-1][1:] if paired["fit"] else None
        row = result_row(record["name"] + " · 最新记录", epoch, fit, evaluation, baseline.get("monitor"), None, None,
                         latest("time/elapsed_seconds"), "原始训练记录")
        profile_id = "native_" + str(record["id"])
        profile = {
            "id": profile_id, "label": record["name"], "kind": "training",
            "native_id": record["id"], "created_at": record["create_time"],
            "status": "running" if record["status"] == 0 else "complete" if record["status"] == 1 else "stopped",
            "native": "/experiment/" + str(record["id"]) + "/chart", "epoch": epoch, "total_epochs": total,
            "progress_text": (str(epoch) + (" / " + str(total) if total else "") + " 轮") if epoch is not None else "等待真实进度记录",
            "updates": latest("progress/updates"),
            "elapsed_seconds": latest("time/elapsed_seconds"), "elapsed_note": "原始日志记录",
            "eta_seconds": latest("time/eta_seconds"), "best_epoch": None,
            "x_title": "Epoch" if dc.get("step_unit") == "epoch" else "原始记录步数",
            "series": paired, "baseline": baseline,
            "checkpoints": {"last": {"row": row, "regions": []}}, "scope_note": dc.get("scope", "尚未声明评测范围"),
            "decision": "等待同口径评测证据", "reason": "只展示真实记录；未配置的 C82 参照、最好权重和耗时不作推断。",
            "evidence": [], "sources": [],
        }
        shared_rows.extend(attach_native_history(profile, cfg, record, history_paths.get(record["run_id"])))
        shared_rows.extend(attach_c148_evidence(profile, cfg, record))
        profiles.append(profile)
    native_db.close()
    phys_rows = []
    for key, label in (("current_solver", "现有原始求解器"), ("sionna_R1", "Sionna · 最多 1 次反射"),
                       ("sionna_R2", "Sionna · 最多 2 次反射"), ("sionna_R3", "Sionna · 最多 3 次反射")):
        model = physics["models"][key]
        phys_rows.append({"label": label, "all": model["all"]["rmse"], "tail": model["tail"]["rmse"]})
    return {
        "schema": 1, "updated_at": datetime.now(timezone.utc).isoformat(), "official_VAL_used": False,
        "records": {"total": len(native), "running": sum(p["status"] == "running" for p in profiles if p["kind"] == "training"),
                    "finished": sum(x["status"] == 1 for x in native)},
        "default_profile": max(
            (p for p in profiles if p["kind"] == "training"),
            key=lambda p: (p["status"] == "running", p.get("native_id", 0)))["id"],
        "profiles": profiles,
        "comparison": {
            "monitor": {"title": "同一地形留出评测 · 628 图 / 128 地形", "rows": shared_rows, "note": scope_note},
            "confirmation": {"title": "确认集 · 1,934 图 / 484 地形", "rows": confirm_rows,
                             "note": "只比较 C146 与该面板 C82。C145 没有此面板记录。此处仍是历史开发数据，非官方 VAL。"},
        },
        "physics": {"rows": phys_rows, "gain_R1_minus_R3_tail": physics["delta_R1_minus_R3"]["tail"],
                    "scope": "8 个诊断地形 / 256 个固定采样点；不与完整地图模型成绩混排。",
                    "note": "增加反射对大误差区的改善很小；高频绝对误差仍受采样影响，未验证替换特征后的学习模型效果。",
                    "native": native_link("C147最终 | Sionna最多3")},
        "sources": {k: {"path": path, "bytes": (ROOT/path).stat().st_size,
                         "modified": datetime.fromtimestamp((ROOT/path).stat().st_mtime, timezone.utc).isoformat(),
                         "url": "/cc/source/" + k} for k,path in SOURCES.items()},
    }

def snapshot():
    global _CACHE
    with _LOCK:
        now = time.monotonic()
        if _CACHE[1] is None or now - _CACHE[0] >= 5:
            value = build()
            # Refuse nonfinite numbers rather than silently making a misleading plot.
            json.dumps(value, allow_nan=False)
            _CACHE = (now, value)
        return _CACHE[1]
