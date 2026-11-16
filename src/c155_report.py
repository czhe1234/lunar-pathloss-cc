"""C155 independent CPU scoring, optimizer audit and terrain-paired comparison."""
from pathlib import Path
import json,os,subprocess,sys,math
import numpy as np
from src.c150_report import load,sha,atomic,evidence,statistics,paired_delta
from src.c155_grazing import score_saved,identity
ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT/"audit/c155_unet";RUN=ROOT/"runs/c155_unet"
BANDS=("415MHz","5p775GHz")

def verify_tag(tag,plan,row):
    marker=RUN/f"{tag}_verified.json"
    if not marker.exists():
        subprocess.run([str(ROOT/".venv/bin/python"),"-u","-m","src.c145_verify",
            "--run",str(RUN.relative_to(ROOT)),"--tag",tag],cwd=ROOT,
            env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"},check=True)
    verified=load(marker);export=load(RUN/f"{tag}_export.json")
    assert verified["status"]=="verified" and not verified["official_VAL_used"]
    assert export["checkpoint_sha256"]==sha(RUN/f"{tag}.pt")
    assert export["net_state_sha256"]==row["net_state_sha256"]
    assert export["model_sha256"]==row["predictor_sha256"]==verified["export_identity"]["model_sha256"]
    contract={"net_state_sha256":row["net_state_sha256"],"output":"registered_FSPL + ConvNeXt_UNet(raw68)",
        "C82_input":False,"C82_output_prior":False,"feature_cache_sha256":plan["cache"]["features_sha256"]}
    import hashlib
    assert hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()==row["predictor_sha256"]
    for name in ("fit","monitor"):
        expected=row["evaluation"][name]
        actual=verified["panels"][name]["groups"]["all"]
        assert np.allclose(expected["rmse"],actual["RMSE_dB"],rtol=0,atol=1e-9)
        assert expected["groups"]["all"]["count"]==actual["valid_pixels"]
        for a,b in zip(("all","tail","under_tail","over_tail","non_tail","C82_good"),
            ("all","C82_abs_error_ge_5dB","C82_underestimated_loss_ge_5dB","C82_overestimated_loss_ge_5dB","C82_abs_error_lt_5dB","C82_abs_error_le_1dB")):
            got=verified["panels"][name]["groups"][b];old=expected["groups"][a]
            assert got["valid_pixels"]==old["count"]
            assert np.allclose(got["SSE_dB2"],old["SSE"],rtol=1e-12,atol=1e-6)
    return verified

def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU report only")
    marker=AUDIT/"comparison_summary.json"
    if marker.exists():
        d=load(marker);assert d["status"]=="verified";print(json.dumps({"event":"reuse_verified_C155_report"}));return
    plan=load(AUDIT/"protocol.json");protocol_sha=sha(AUDIT/"protocol.json")
    for n,h in plan["source_sha256"].items():assert sha(ROOT/n)==h,n
    for n,v in plan["protected_file_identities"].items():assert identity(ROOT/n)==v,n
    for n,v in plan["cache_file_identities"].items():assert identity(ROOT/n)==v,n
    result=load(RUN/"result.json");cfg=load(RUN/"config.json");hist=load(RUN/"history.json")
    assert cfg["protocol"]==plan and cfg["protocol_sha256"]==result["protocol_sha256"]==protocol_sha
    assert result["status"]=="complete" and result["epochs"]==60 and result["updates"]==21120
    assert [h["epoch"] for h in hist]==[0,*range(4,61,4)]
    assert all(h["updates"]==352*h["epoch"] for h in hist)
    selected=min(hist,key=lambda h:float(np.mean(h["evaluation"]["monitor"]["rmse"])))
    assert selected["epoch"]==result["selected_epoch"]
    verified={}
    for tag,row in (("selected",selected),("last",hist[-1])):
        verified[tag]=verify_tag(tag,plan,row)
    current=statistics(RUN/"selected_terrain_sse.npz",plan["panels"],verified["selected"])
    control_verified=load(ROOT/"runs/c153_standalone_c151a/selected_verified.json")
    control=statistics(ROOT/"runs/c153_standalone_c151a/selected_terrain_sse.npz",plan["panels"],control_verified)
    assert np.array_equal(current["monitor"]["terrain_names"],control["monitor"]["terrain_names"])
    assert np.array_equal(current["monitor"]["terrain_count"],control["monitor"]["terrain_count"])
    primary=paired_delta(current["monitor"]["terrain_sse"],control["monitor"]["terrain_sse"],current["monitor"]["terrain_count"])
    c82=paired_delta(current["monitor"]["terrain_sse"],current["monitor"]["terrain_C82_sse"],current["monitor"]["terrain_count"])
    grazing={}
    for tag,row in (("selected",selected),("last",hist[-1])):
        path=RUN/f"{tag}_grazing_verified.json"
        if path.exists():v=load(path)
        else:
            v=score_saved(RUN,tag);atomic(path,v)
        for name in ("fit","monitor"):
            g=row["evaluation"][name]["grazing"]
            for k in ("count","GT_minus_FSPL_highpass_RMS_dB","Pred_minus_FSPL_highpass_RMS_dB","highpass_error_RMS_dB","localmean_error_RMS_dB"):
                assert np.allclose(v[name][k],g[k],rtol=0,atol=1e-9),(tag,name,k,v[name][k],g[k])
        grazing[tag]=v
    import torch,hashlib
    cp=torch.load(RUN/"last.pt",map_location="cpu",weights_only=False)
    assert cp["epoch"]==60 and cp["updates"]==21120 and cp["protocol_sha256"]==protocol_sha
    assert cp["scheduler"]=={"kind":"linear_warmup_cosine_formula","updates":21120,"total_updates":21120}
    assert {int(v["step"].item()) for v in cp["optimizer"]["state"].values()}=={21120}
    assert all(math.isclose(g["lr"],1e-5,rel_tol=1e-12) for g in cp["optimizer"]["param_groups"])
    rng=torch.Generator().manual_seed(plan["data_seed"])
    for epoch in range(60):torch.randperm(len(plan["panels"]["fit"]),generator=rng)
    assert torch.equal(cp["order_generator"],rng.get_state())
    assert not torch.cuda.is_initialized()
    print(json.dumps({"event":"C155_predictions_Grazing_optimizer_and_order_verified"}),flush=True)
    for n,h in plan["protected_readonly_sha256"].items():assert sha(ROOT/n)==h,n
    assert all(identity(ROOT/n)==v for n,v in plan["cache_file_identities"].items())
    supports=load(AUDIT/"grazing_support.json")
    for name in ("fit","monitor"):
        assert sha(ROOT/supports["supports"][name]["path"])==supports["supports"][name]["sha256"]
    baseline=supports["C153_selected12"]
    gaps={}
    for tag,row in (("selected",selected),("last",hist[-1])):
        gaps[tag]=(np.asarray(row["evaluation"]["monitor"]["rmse"])-np.asarray(row["evaluation"]["fit"]["rmse"])).tolist()
    out={"status":"verified","campaign":"C155","protocol_sha256":protocol_sha,"epochs":60,"updates":21120,
        "parameters":plan["parameters"],"selected_epoch":selected["epoch"],
        "selected":selected["evaluation"],"last":hist[-1]["evaluation"],"C153_selected12":baseline,
        "C153_last":plan["control_last"],"generalization_gap_dB":gaps,
        "C153_last_gap_dB":(np.asarray(plan["control_last"]["monitor"])-np.asarray(plan["control_last"]["FIT"])).tolist(),
        "paired_comparison_C153_selected12":primary,"paired_comparison_C82":c82,
        "both_band_improvement_vs_C153_with95CI":primary["both_band_upper_CI_strictly_below_zero"],
        "both_band_improvement_vs_C82_with95CI":c82["both_band_upper_CI_strictly_below_zero"],
        "Grazing_independent_CPU":grazing,"optimizer_and_data_order_verified":True,
        "protected_C82_C145_C153_SHA_unchanged":True,"old_experiments_rerun":False,"report_network_inference":False,
        "automatic_extension":False,"official_VAL_used":False,"source_sha256":sha(Path(__file__)),
        "limits":[plan["comparison_limit"],"One architecture replacement changes capacity, receptive field and inductive bias together; it cannot isolate parameter count or attention alone.",
            "C153 full historical Grazing curves were not saved; only its selected12 saved-prediction highpass comparator is available. No old inference rerun."]}
    from src import cc_tracking
    swan=cc_tracking.init(name="C155 verified architecture comparison",campaign="C155",
        config={"run_kind":"analysis.paired_comparison","_cc_dashboard":{"kind":"analysis"},"protocol_sha256":protocol_sha},
        description="CPU selected/final scores and Grazing checks;128terrain paired comparison versus C153/C82.",tags=["CPU","paired-terrain","unet"])
    metrics={"selection/epoch":selected["epoch"],"gate/vs_C153":int(out["both_band_improvement_vs_C153_with95CI"]),
        "gate/vs_C82":int(out["both_band_improvement_vs_C82_with95CI"])}
    for j,b in enumerate(BANDS):
        for ref,pair in (("C153",primary),("C82",c82)):
            metrics[f"{ref}/{b}/delta_RMSE_dB"]=pair["point_delta_RMSE_dB"][j]
            metrics[f"{ref}/{b}/CI_low"]=pair["delta_RMSE_dB_95CI"][j][0]
            metrics[f"{ref}/{b}/CI_high"]=pair["delta_RMSE_dB_95CI"][j][1]
    cc_tracking.log(metrics,step=1);out["tracking"]={"run_id":swan.id,"dir":str(swan.dir)};cc_tracking.finish()
    atomic(marker,out)
    lines=["# C155轻量ConvNeXt-UNet对照","",f"完整60轮/21120更新，{plan['parameters']:,}参数，共同选中Epoch{selected['epoch']}。","",
        "|模型/轮次|FIT415|FIT5775|monitor415|monitor5775|","|---|---:|---:|---:|---:|"]
    for label,f,m in (("C153 selected12",baseline["fit"]["rmse"],baseline["monitor"]["rmse"]),
        ("C155 selected"+str(selected["epoch"]),selected["evaluation"]["fit"]["rmse"],selected["evaluation"]["monitor"]["rmse"]),
        ("C153 last60",plan["control_last"]["FIT"],plan["control_last"]["monitor"]),
        ("C155 last60",hist[-1]["evaluation"]["fit"]["rmse"],hist[-1]["evaluation"]["monitor"]["rmse"])):
        lines.append("|"+label+"|"+"|".join(f"{x:.6f}" for x in [*f,*m])+"|")
    lines+=["","相对C153 selected12的128地形配对95%CI："+str(primary),"",
        "Grazing高频误差：","|模型/面板|415MHz|5.775GHz|","|---|---:|---:|"]
    for name in ("fit","monitor"):
        for label,g in (("C153 selected12",baseline[name]["grazing"]),("C155 selected",grazing["selected"][name]),("C155 last60",grazing["last"][name])):
            lines.append(f"|{label} {name}|"+ "|".join(f"{x:.6f}" for x in g["highpass_error_RMS_dB"])+"|")
    lines+=["",*out["limits"],"","冻结预算结束，不自动加训。"]
    (AUDIT/"comparison_summary.md").write_text("\n".join(lines)+"\n")
    status=load(AUDIT/"controller_status.json");status.update(stage="complete",status="complete",independent_CPU_verified=True,
        comparison_summary=str(marker.relative_to(ROOT)),selected_epoch=selected["epoch"])
    atomic(AUDIT/"controller_status.json",status)
    print(json.dumps({"status":"verified","selected_epoch":selected["epoch"],"monitor":selected["evaluation"]["monitor"]["rmse"],
        "vs_C153":primary,"vs_C82":c82}),flush=True)
if __name__=="__main__":main()
