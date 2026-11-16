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
    for band,band_title in (("415MHz","415 MHz"),("5p775GHz","5.775 GHz")):
        for panel,panel_title in (("fit","训练地形"),("monitor","本次留出地形")):
            for metric,title in (("Grazing_HP_error_RMS","Grazing 高频误差 RMS"),
                ("Grazing_GT_HP_RMS","Grazing GT 高频 RMS"),("Grazing_Pred_HP_RMS","Grazing Pred 高频 RMS"),
                ("Grazing_localmean_error_RMS","Grazing 局部均值误差 RMS")):
                names[f"{panel}/{metric}_{band}_dB"]=f"{panel_title}：{title}（{band_title}，dB）"
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

def paired_rmse_patch(content):
    """Enable native multi-metric scalar charts through chart view metadata only."""
    import re
    prefix = r"""/* CC_DUAL_RMSE_BEGIN */
function ccDualChartConfig(chart){try{const c=typeof chart.config==="string"?JSON.parse(chart.config):chart.config;return c&&c.cc_dual_rmse||null}catch{return null}}
function ccDualMetricColor(name){return String(name).startsWith("fit/")?"#26836d":"#397abb"}
function ccDualNativeOptions(chart,options){const c=ccDualChartConfig(chart);if(!c)return options;return {...options,seriesField:"series",smooth:false,annotations:[{type:"line",start:["min",c.baseline],end:["max",c.baseline],style:{stroke:"#85978e",lineWidth:1.5,lineDash:[6,4]},text:{content:(c.baseline_label||"C82")+" "+Number(c.baseline).toFixed(4)+" dB",position:"end",offsetY:-8,style:{fill:"#62736b",fontSize:11,textAlign:"right"}}}],yAxis:{min:null,title:{text:c.y_title||"RMSE (dB)"},label:{formatter:v=>Number(v).toFixed(2)},grid:{line:{style:{stroke:"#e6ece9"}}}}}}
/* CC_DUAL_RMSE_END */
"""
    content = re.sub(r"/\* CC_DUAL_RMSE_BEGIN \*/.*?/\* CC_DUAL_RMSE_END \*/\n?", "", content, flags=re.S)
    replacements = {
        's=r.chart.multi,l=p("chartRefList")': 's=r.chart.multi||!!ccDualChartConfig(r.chart),l=p("chartRefList")',
        'const f=p("getColor"),d=p("defaultColor")': 'const ccNativeColor=p("getColor"),f=(t,e,n)=>ccDualChartConfig(r.chart)?ccDualMetricColor(t):ccNativeColor(t,e,n),d=p("defaultColor")',
        '{xField:M,xTitle:S}=sH[Q],H=': '{xField:M,xTitle:ccNativeAxisTitle}=sH[Q],S=ccDualChartConfig(r.chart)?"Epoch":ccNativeAxisTitle,H=',
        'smooth:!1,animation:!1,...n});return o.render()': 'smooth:!1,animation:!1,...ccDualNativeOptions(r.chart,n)});return o.render()',
    }
    for old,new in replacements.items():
        if new not in content:
            assert content.count(old)==1, "Native line renderer changed: "+old
            content=content.replace(old,new,1)
    return prefix+content

def configure_paired_rmse(experiment_id, profile=None, grazing=False):
    """Pin two genuine SwanBoard charts linked to existing tags; preserve all logs."""
    from datetime import datetime,timezone
    db=sqlite3.connect(STATE/"logs/runs.swanlab",timeout=30)
    db.row_factory=sqlite3.Row
    db.execute("pragma busy_timeout=30000")
    experiment=db.execute("select * from experiment where id=?",(experiment_id,)).fetchone()
    assert experiment is not None
    namespaces=list(db.execute("select * from namespace where experiment_id=?",(experiment_id,)))
    assert namespaces
    rootns=next((x for x in namespaces if x["name"]=="default"),namespaces[0])
    kind="grazing" if grazing else "rmse"
    backup=VIEW/f"paired_{kind}_before_{experiment_id}.json"
    if not backup.exists():
        snapshot={"experiment":dict(experiment),"charts":[dict(x) for x in db.execute("select * from chart where experiment_id=?",(experiment_id,))],
            "namespaces":[dict(x) for x in namespaces],
            "source_count":db.execute("select count(*) from source").fetchone()[0],
            "tag_rows":[dict(x) for x in db.execute("select * from tag where experiment_id=?",(experiment_id,))]}
        backup.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n")
    # Use the already verified baseline metadata of the selected training profile.
    if profile is None and not grazing:
        from urllib.request import urlopen
        data=json.load(urlopen("http://127.0.0.1:16007/cc/data.json",timeout=30))
        profile=next(x for x in data["profiles"] if x.get("native")==f"/experiment/{experiment_id}/chart")
    if grazing:
        support=json.loads((ROOT/"audit/c155_unet/grazing_support.json").read_text())
        baselines=support["C153_selected12"]["monitor"]["grazing"]["highpass_error_RMS_dB"]
    else:baselines=profile["baseline"]["monitor"]
    assert len(baselines)==2 and all(isinstance(x,(int,float)) for x in baselines)
    now=datetime.now(timezone.utc).isoformat()
    ids=[];sources=[]
    db.execute("begin immediate")
    try:
        if grazing:
            nsname="Grazing 高频误差（展开查看）"
            rootns=db.execute("select * from namespace where experiment_id=? and name=?",(experiment_id,nsname)).fetchone()
            if rootns is None:
                cursor=db.execute("insert into namespace(experiment_id,project_id,name,description,sort,opened,more,create_time,update_time) values(?,NULL,?,?,1,0,NULL,?,?)",
                    (experiment_id,nsname,"9点均有效且均为Grazing；高频误差越低越好；灰虚线为C153 selected12 monitor",now,now))
                rootns=db.execute("select * from namespace where id=?",(cursor.lastrowid,)).fetchone()
        for j,(key,bandtitle) in enumerate((("415MHz","415 MHz"),("5p775GHz","5.775 GHz"))):
            title=bandtitle+(" · Grazing 高频误差 (dB)" if grazing else " · RMSE (dB)")
            metric="Grazing_HP_error_RMS" if grazing else "RMSE"
            names=[f"fit/{metric}_{key}_dB",f"monitor/{metric}_{key}_dB"]
            tags=[db.execute("select * from tag where experiment_id=? and name=?",(experiment_id,n)).fetchone() for n in names]
            assert all(t is not None and t["type"]=="line" for t in tags),names
            view=json.dumps({"cc_dual_rmse":{"band":key,"baseline":baselines[j],"source":"existing native scalar tags",
                "colors":{"fit":"#26836d","monitor":"#397abb","C82":"#85978e"},"metric_names":names,
                "baseline_label":"C153 selected12 monitor" if grazing else "C82",
                "y_title":"高频误差 RMS (dB)" if grazing else "RMSE (dB)"}},ensure_ascii=False)
            old=db.execute("select * from chart where experiment_id=? and name=?",(experiment_id,title)).fetchone()
            if old:
                assert old["system"]==0 and old["config"] and "cc_dual_rmse" in old["config"]
                chart_id=old["id"]
                db.execute("update chart set status=?,sort=?,config=?,update_time=? where id=?",(0 if grazing else 1,j,view,now,chart_id))
            else:
                cursor=db.execute("insert into chart(experiment_id,project_id,name,description,system,type,reference,status,sort,config,more,create_time,update_time) values(?,NULL,?,?,0,'line','step',1,?,?,NULL,?,?)",
                    (experiment_id,title,"训练地形、留出地形的原始RMSE；灰虚线为同面板C82",j,view,now,now))
                chart_id=cursor.lastrowid
            for k,tag in enumerate(tags):
                if not db.execute("select id from source where chart_id=? and tag_id=?",(chart_id,tag["id"])).fetchone():
                    db.execute("insert into source(tag_id,chart_id,sort,error,more,create_time,update_time) values(?,?,?,'','',?,?)",
                        (tag["id"],chart_id,k,now,now))
            if not db.execute("select id from display where chart_id=? and namespace_id=?",(chart_id,rootns["id"])).fetchone():
                db.execute("insert into display(chart_id,namespace_id,sort,more,create_time,update_time) values(?,?,?,'',?,?)",
                    (chart_id,rootns["id"],j,now,now))
            if grazing:db.execute("update chart set status=0 where id=?",(chart_id,))
            ids.append(chart_id);sources.append({"chart_id":chart_id,"metric_names":names,"tag_ids":[x["id"] for x in tags],
                "reference_label":"C153 selected12 monitor" if grazing else "C82","reference":baselines[j]})
        if not grazing:db.execute("update chart set status=-1,sort=id+10000 where experiment_id=? and id not in (?,?)",(experiment_id,*ids))
        db.execute("update namespace set opened=0 where experiment_id=?",(experiment_id,))
        db.execute("update experiment set pinned_opened=1,hidden_opened=0 where id=?",(experiment_id,))
        db.commit()
    except BaseException:
        db.rollback();raise
    after=[dict(x) for x in db.execute("select * from tag where experiment_id=?",(experiment_id,))]
    assert after==json.loads(backup.read_text())["tag_rows"], "Raw tag metadata changed"
    db.close()
    result={"experiment_id":experiment_id,"chart_ids":ids,"source_links":sources,
        "original_training_metrics_configs_and_artifacts_unchanged":True,
        "other_charts":"hidden and collapsed, retained","native_multi_metric_renderer":True}
    (VIEW/f"paired_{kind}_{experiment_id}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-layout", action="store_true",
                        help="Pin key plots and collapse complete historical statistics.")
    parser.add_argument("--paired-rmse", action="store_true")
    parser.add_argument("--paired-grazing", action="store_true")
    parser.add_argument("--experiment-id", type=int)
    parser.add_argument("--paired-all-training", action="store_true")
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
    content = paired_rmse_patch(content)
    if content != before:
        temporary=asset.with_suffix(asset.suffix+".cc.partial")
        temporary.write_text(content);temporary.replace(asset)
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
    if args.paired_rmse:
        assert args.experiment_id is not None
        report["paired_rmse"] = configure_paired_rmse(args.experiment_id)
    if args.paired_grazing:
        assert args.experiment_id is not None
        report["paired_grazing"]=configure_paired_rmse(args.experiment_id,grazing=True)
    if args.paired_all_training:
        from urllib.request import urlopen
        data=json.load(urlopen("http://127.0.0.1:16007/cc/data.json",timeout=30))
        report["paired_training"]=[]
        for profile in data["profiles"]:
            if profile.get("kind")=="training" and profile.get("native","").startswith("/experiment/"):
                experiment_id=int(profile["native"].split("/")[2])
                report["paired_training"].append(configure_paired_rmse(experiment_id,profile))
    print(json.dumps(report, ensure_ascii=False))
if __name__ == "__main__":
    main()
