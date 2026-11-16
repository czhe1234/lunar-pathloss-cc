"""C153 independent CPU verification of saved selected predictions; no inference."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c150_report import load,sha,atomic,clean,evidence,statistics,paired_delta
AUDIT=ROOT/"audit/c153_standalone_c151a"
RUN=ROOT/"runs/c153_standalone_c151a"

def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU report requires empty CUDA_VISIBLE_DEVICES")
    plan=load(AUDIT/"protocol.json");identity=sha(AUDIT/"protocol.json")
    if (AUDIT/"comparison_summary.json").exists():
        old=load(AUDIT/"comparison_summary.json")
        assert old["status"]=="verified" and old["protocol_sha256"]==identity
        print(json.dumps({"status":"reused_existing_verified_result","selected_epoch":old["selected_epoch"]}));return
    for f,h in plan["source_sha256"].items():assert sha(ROOT/f)==h,f
    for f,h in plan["protected_readonly_sha256"].items():assert sha(ROOT/f)==h,f
    assert sha(ROOT/plan["stage1"]["path"])==plan["stage1"]["sha256"]
    cfg=load(RUN/"config.json");result=load(RUN/"result.json")
    hist=load(RUN/"history.json");export=load(RUN/"selected_export.json")
    assert cfg["protocol"]==plan and cfg["protocol_sha256"]==result["protocol_sha256"]==identity
    assert result["status"]=="complete" and result["epochs"]==60 and result["updates"]==21120
    assert [v["epoch"] for v in hist]==[0,*range(4,61,4)]
    assert all(v["updates"]==v["epoch"]*352 for v in hist)
    selected=min(hist,key=lambda v:float(np.mean(v["evaluation"]["monitor"]["rmse"])))
    assert selected["epoch"]==result["selected_epoch"]==export["training_state"]["epoch"]
    assert selected["predictor_sha256"]==result["selected_predictor_sha256"]==export["model_sha256"]
    assert export["net_state_sha256"]==selected["net_state_sha256"]
    assert export["checkpoint_sha256"]==sha(RUN/"selected.pt")
    assert np.allclose(selected["evaluation"]["monitor"]["rmse"],result["selected_monitor_RMSE_dB"],rtol=0,atol=1e-12)
    context={"directory":RUN,"config":cfg,"result":result,"history":hist,"selected":selected,"export":export,"arm":"standalone"}
    verified=evidence(context,hash_predictions=True)
    if verified is None:
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
        subprocess.run([str(ROOT/".venv/bin/python"),"-u","-m","src.c145_verify",
            "--run",str(RUN.relative_to(ROOT)),"--tag","selected"],cwd=ROOT,env=env,check=True)
        verified=evidence(context,hash_predictions=True)
    assert verified is not None
    current=statistics(RUN/"selected_terrain_sse.npz",plan["panels"],verified)
    st=current["monitor"]
    pair=paired_delta(st["terrain_sse"],st["terrain_C82_sse"],st["terrain_count"])
    assert pair["terrains"]==128 and pair["replicates"]==2000 and pair["seed"]==145073
    out={"status":"verified","campaign":"C153","protocol_sha256":identity,"epochs":60,"updates":21120,
        "selected_epoch":selected["epoch"],
        "selected":{p:verified["panels"][p]["groups"] for p in ("fit","monitor")},
        "last_from_training_evaluation":hist[-1]["evaluation"],"paired_comparison_C82":pair,
        "gate_passed":pair["both_band_upper_CI_strictly_below_zero"],
        "automatic_budget_extension":False,"official_VAL_used":False,
        "training_or_network_inference_run_by_report":False,
        "frozen_C82_C145_artifacts_unchanged":True,
        "evidence":{"selected_verified_sha256":sha(RUN/"selected_verified.json"),
            "selected_terrain_sse_sha256":sha(RUN/"selected_terrain_sse.npz"),
            "predictor_sha256":export["model_sha256"],"checkpoint_sha256":export["checkpoint_sha256"],
            "report_source_sha256":sha(Path(__file__))},
        "limits":[plan["comparison_limit"],"Fresh one-seed60epoch schedule differs from old12epoch development candidates; report performance without assigning a single causal factor."]}
    marker=AUDIT/"comparison_tracking.json"
    if not marker.exists():
        from src import cc_tracking
        swan=cc_tracking.init(name="C153 verified standalone comparison",campaign="C153",
            config={"run_kind":"analysis.paired_comparison","_cc_dashboard":{"kind":"analysis"},"protocol_sha256":identity},
            description="CPU saved-prediction verification and128terrain paired95CI vs frozenC82.",tags=["CPU","paired-terrain"])
        values={"selection/epoch":out["selected_epoch"],"gate/passed":int(out["gate_passed"])}
        for j,b in enumerate(("415MHz","5p775GHz")):
            values[f"C82/{b}/delta_RMSE_dB"]=pair["point_delta_RMSE_dB"][j]
            values[f"C82/{b}/CI_low"]=pair["delta_RMSE_dB_95CI"][j][0]
            values[f"C82/{b}/CI_high"]=pair["delta_RMSE_dB_95CI"][j][1]
        cc_tracking.log(values,step=1);track={"run_id":swan.id,"dir":str(swan.dir)};cc_tracking.finish();atomic(marker,track)
    out["tracking"]=load(marker);atomic(AUDIT/"comparison_summary.json",clean(out))
    lines=["# C153独立基线：已完成60轮并独立CPU核验","",
        f"选中同一Epoch {out['selected_epoch']}，共21120更新。输出registered_FSPL+Field；C82输入槽位为零。",
        "","|频段|C82 RMSE dB|C153 RMSE dB|C153-C82 dB|配对95%CI dB|",
        "|---|---:|---:|---:|---|"]
    for j,b in enumerate(("415MHz","5p775GHz")):
        v=out["selected"]["monitor"]["all"]
        lines.append(f"|{b}|{v['C82_RMSE_dB'][j]:.9f}|{v['RMSE_dB'][j]:.9f}|{pair['point_delta_RMSE_dB'][j]:+.9f}|{pair['delta_RMSE_dB_95CI'][j]}|")
    lines+=["",f"双频门槛通过：{out['gate_passed']}。固定预算结束，不自动加训。","",*out["limits"]]
    (AUDIT/"comparison_summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"verified","selected_epoch":out["selected_epoch"],"gate_passed":out["gate_passed"],"paired_comparison_C82":pair}),flush=True)
if __name__=="__main__":main()
