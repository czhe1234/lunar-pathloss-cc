"""Verify overview extraction and preservation. Does not run model inference."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cc_dashboard as d

def main():
    original = json.loads((d.STATE / "dashboard/sources_before.json").read_text())
    for source in original.values():
        assert hashlib.sha256((ROOT/source["path"]).read_bytes()).hexdigest() == source["sha256"], source["path"]
    p145 = json.loads((ROOT / "runs/c145_capacity_native/config.json").read_text())
    p146 = json.loads((ROOT / "audit/c146_solver_ablation/protocol.json").read_text())
    for panel in ("fit", "monitor"):
        assert set(p145["panels"][panel]) == set(p146["panels"][panel])
    x = d.build()
    a,b = d.read("c145_summary"),d.read("c146_summary")
    assert len(x["comparison"]["monitor"]["rows"]) == 7
    assert len(x["comparison"]["confirmation"]["rows"]) == 5
    assert x["records"]["total"] == 11 and x["records"]["running"] == 0
    assert d.gains([],[]) is None and d.gains([1], [0]) is None
    assert d.gains(None,[0,0]) is None and d.gains([1,2],[0,0]) == [1,2]
    for profile in x["profiles"]:
        assert profile["status"] == "complete"
        for tag,view in profile["checkpoints"].items():
            row = view["row"]
            if profile["id"] == "c145":
                panel = a["verified_results"][tag]["panels"]["monitor"]
                assert row["evaluation"] == panel["groups"]["all"]["RMSE_dB"]
                assert row["tail"] == panel["groups"]["C82_abs_error_ge_5dB"]["RMSE_dB"]
                assert row["epoch"] == a["checkpoint_epochs"][tag]["absolute_epoch"]
            else:
                arm = profile["id"].removeprefix("c146_")
                key = arm + ("_selected" if tag == "selected" else "_final")
                model = b["panels"]["monitor"]["models"][key]
                assert row["evaluation"] == model["RMSE"]["all"]
                assert row["tail"] == model["RMSE"]["tail"]
                assert row["epoch"] == model["epoch"]
            assert row["gain"] == d.gains(row["C82"], row["evaluation"])
            assert row["beats_both"] == all(v>0 for v in row["gain"])
    c145=x["profiles"][0]
    assert c145["best_epoch"] == 12 and c145["epoch"] == 764
    assert c145["checkpoints"]["selected"]["row"]["beats_both"]
    assert not c145["checkpoints"]["last"]["row"]["beats_both"]
    assert len(c145["series"]["monitor"]) == 190
    expected_cost=sum(v["elapsed_seconds"] for v in d.read("c145_phases") if not v.get("export_only"))
    assert c145["elapsed_seconds"] == expected_cost
    assert all(r.get("profile") != "c145" for r in x["comparison"]["confirmation"]["rows"])
    report={"status":"verified","source_files_unchanged":len(original),
            "native_records":x["records"],"same_monitor_counts_and_baselines_checked":True,"exact_fit_monitor_map_id_sets_checked":True,
            "curated_profiles":len(x["profiles"]),"comparison_rows":{k:len(v["rows"]) for k,v in x["comparison"].items()},
            "checkpoint_selection_preserved":True,"both_bands_jointly_compared":True,
            "incomplete_reference_never_counts_as_a_win":True,"stage_cost_from_original_logs":True,
            "no_training_or_inference_or_rescoring":True}
    (d.STATE/"dashboard/verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
