"""Reduce offline SwanBoard rendering without changing any experiment records."""
import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "tracking/swanlab"
VIEW = STATE / "view"


def chinese_label_patch(content):
    """Translate visible labels only; keep metric keys and API sources unchanged."""
    import re
    names = {}
    for band, chinese in (("415MHz", "415 MHz"), ("5p775GHz", "5.775 GHz")):
        for prefix, label in (
            ("fit/RMSE_", "训练过的地形：RMSE"),
            ("monitor/RMSE_", "本次训练留出的地形：RMSE"),
            ("fit/tail/RMSE_", "训练地形中原来容易错的位置：RMSE"),
            ("monitor/tail/RMSE_", "留出地形中原来容易错的位置：RMSE"),
            ("train/running_weighted_RMSE_", "训练过程中的加权 RMSE"),
        ):
            names[prefix + band + "_dB"] = label + "（" + chinese + "，dB）"
        for region, label in (
            ("all", "全部采样点"), ("tail", "原来容易错的位置"),
            ("common_unfloored", "各求解器都有效的采样点"),
            ("tail_common_unfloored", "各求解器都有效的大误差采样点"),
        ):
            names["aggregate/" + region + "/rmse_" + band] = label + "：RMSE（" + chinese + "，dB）"
    names.update({
        "learning_rate": "学习率：每次调整参数的步幅",
        "optim/learning_rate": "学习率：每次调整参数的步幅",
        "evaluation/fit/weighted_MSE_objective": "训练集加权误差目标",
        "migration/artifact_index": "原始结果文件目录",
        "migration/report": "原始实验报告",
    })
    tokens = {
        "fit": "训练过的地形", "monitor": "本次训练留出的地形",
        "confirmation": "确认用的开发数据", "all": "全部位置",
        "tail": "原来容易错的位置", "under_tail": "原来低估损耗的位置",
        "over_tail": "原来高估损耗的位置", "non_tail": "其他位置",
        "C82_good": "C82原来较准的位置", "groups": "分组结果", "evaluation": "评测",
        "verified": "已核对的结果", "selected": "最佳保存版本", "final": "结束时的版本",
        "last": "结束时的版本", "aggregate": "汇总", "per_map": "逐地图",
        "RMSE": "RMSE（预测误差）", "rmse": "RMSE（预测误差）", "C82_RMSE": "C82 RMSE",
        "SSE": "SSE（平方误差总和）", "count": "计分位置数量",
        "progress": "进度", "epoch": "训练轮数", "absolute_epoch": "总训练轮数",
        "additional_epoch": "本次新增训练轮数", "additional_step": "本次参数更新次数",
        "time": "耗时", "elapsed_seconds": "已用秒数",
        "best": "最佳记录", "train": "训练过程",
    }
    prefix = "/* CC_CHINESE_LABELS_BEGIN */\n"
    prefix += "const ccCnNames=" + json.dumps(names, ensure_ascii=False) + ";\n"
    prefix += "const ccCnTokens=" + json.dumps(tokens, ensure_ascii=False) + ";\n"
    prefix += 'function ccCnMetricLabel(value){const s=String(value??"");if(ccCnNames[s])return ccCnNames[s];return s.split("/").map(x=>ccCnTokens[x]||x).join(" / ").replace(/without_solver/g,"不使用求解器信息").replace(/with_solver/g,"使用求解器信息").replace(/415MHz/g,"415 MHz").replace(/5p775GHz/g,"5.775 GHz").replace(/_dB/g," (dB)");}\n'
    prefix += "/* CC_CHINESE_LABELS_END */\n"
    content = re.sub(r"/\* CC_CHINESE_LABELS_BEGIN \*/.*?/\* CC_CHINESE_LABELS_END \*/\n?", "", content, flags=re.S)
    content = content.replace('xTitle:"记录进度"', 'xTitle:"Step"').replace('default:"记录进度"', 'default:"Step"')
    substitutions = {
        "e.name.toLowerCase().includes(t.toLowerCase())": "(e.name+\" \"+ccCnMetricLabel(e.name)).toLowerCase().includes(t.toLowerCase())",
        "P(e.title)": "P(ccCnMetricLabel(e.title))",
        "P(t.data.series)": "P(ccCnMetricLabel(t.data.series))",
        "N(P(t.name),1)]),_:2},1032,[\"to\"])": "N(P(ccCnMetricLabel(t.name)),1)]),_:2},1032,[\"to\"])",
    }
    for old, new in substitutions.items():
        if new not in content:
            assert old in content, "SwanBoard display code changed: " + old
            content = content.replace(old, new)
    return prefix + content

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-layout", action="store_true",
                        help="Pin key plots and collapse complete historical statistics.")
    args = parser.parse_args()
    assert importlib.metadata.version("swanboard") == "0.1.10b2", "Recheck UI patch for the new SwanBoard version"
    package = Path(importlib.util.find_spec("swanboard").origin).parent
    assets = list((package / "template/assets").glob("chart-*.js"))
    assert len(assets) == 1, assets
    asset = assets[0]
    old = "H(e.charts,(t,n)=>(I(),X(f_,"
    new = "H(o.value?e.charts:[],(t,n)=>(I(),X(f_,"
    content = asset.read_text()
    before = content
    VIEW.mkdir(parents=True, exist_ok=True)
    if new not in content:
        assert content.count(old) == 1, "SwanBoard frontend changed; refusing an unverified patch"
        backup = VIEW / (asset.name + ".original")
        if not backup.exists():
            shutil.copy2(asset, backup)
        content = content.replace(old, new, 1)
    assert new in content
    content = chinese_label_patch(content)
    if content != before:
        asset.write_text(content)
    report = {"lazy_mount_closed_groups": True, "chinese_visible_labels": True, "asset": str(asset), "layout_applied": args.apply_layout}
    if args.apply_layout:
        db = sqlite3.connect(STATE / "logs/runs.swanlab")
        db.row_factory = sqlite3.Row
        before_file = VIEW / "layout_before.json"
        if not before_file.exists():
            before = {
                "chart": [dict(x) for x in db.execute("select id,status,sort from chart")],
                "namespace": [dict(x) for x in db.execute("select id,opened from namespace")],
                "experiment": [dict(x) for x in db.execute("select id,pinned_opened,hidden_opened from experiment")],
                "project": [dict(x) for x in db.execute("select id,pinned_opened,hidden_opened from project")],
            }
            before_file.write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n")
        names = []
        for prefix in ("fit/RMSE", "monitor/RMSE", "fit/tail/RMSE", "monitor/tail/RMSE",
                       "train/running_weighted_RMSE"):
            names += [prefix + "_415MHz_dB", prefix + "_5p775GHz_dB"]
        names += ["learning_rate", "optim/learning_rate", "evaluation/fit/weighted_MSE_objective"]
        for region in ("all", "tail", "common_unfloored", "tail_common_unfloored"):
            names += [f"aggregate/{region}/rmse_415MHz", f"aggregate/{region}/rmse_5p775GHz"]
        priority = {name: i for i, name in enumerate(names)}
        summary = {}
        for chart in db.execute("select id,experiment_id,project_id,name from chart").fetchall():
            pinned = chart["name"] in priority
            db.execute("update chart set status=?,sort=? where id=?",
                       (1 if pinned else 0, priority.get(chart["name"]), chart["id"]))
            key = "experiment:" + str(chart["experiment_id"]) if chart["experiment_id"] else "project:" + str(chart["project_id"])
            row = summary.setdefault(key, {"key_charts": 0, "collapsed_charts": 0})
            row["key_charts" if pinned else "collapsed_charts"] += 1
        db.execute("update namespace set opened=0")
        db.execute("update experiment set pinned_opened=1,hidden_opened=0")
        db.execute("update project set pinned_opened=1,hidden_opened=0")
        db.commit()
        db.close()
        report["layout"] = summary
        report["all_metric_values_steps_configs_artifacts_preserved"] = True
        (VIEW / "configuration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
if __name__ == "__main__":
    main()
