"""Verify imported records against immutable source snapshots; no scientific evaluation."""
import hashlib,json,sqlite3,subprocess
from collections import Counter
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/"tracking/swanlab/migration"
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((M/"manifest.json").read_text())
    bundle=json.loads((M/"source_export.json").read_text())
    sources={"mlflow:"+r["info"]["run_uuid"]:{"config":r["params"],"metrics":r["metrics"]} for r in bundle["runs"]}
    c145=json.loads((M/"source_c145.json").read_text())
    sources["raw_json:"+c145["source_run_id"]]=c145
    db=sqlite3.connect("file:"+str(ROOT/"tracking/swanlab/logs/runs.swanlab")+"?mode=ro",uri=True)
    db.row_factory=sqlite3.Row
    experiments={r["run_id"]:dict(r) for r in db.execute("select * from experiment")}
    errors=[];results=[];total=0;artifacts=0
    for key,run in manifest["runs"].items():
        src=sources[key];folder=ROOT/run["run_dir"]
        exp=experiments.get(folder.name)
        if exp is None:
            errors.append({"run":key,"missing_native_run":folder.name});continue
        expected=Counter((x["key"],int(x["step"]),float(x["value"])) for x in src["metrics"])
        actual=Counter()
        for tag in db.execute("select name,folder,type from tag where experiment_id=?",(exp["id"],)):
            if tag["type"]!="line":continue
            for f in (folder/"logs"/tag["folder"]).glob("*.log"):
                for line in f.read_text().splitlines():
                    x=json.loads(line)
                    actual[(tag["name"],int(x["index"]),float(x["data"]))]+=1
        missing=expected-actual;extra=actual-expected
        if missing or extra:
            errors.append({"run":key,"missing":list(missing.items())[:8],"extra":list(extra.items())[:8],
                           "expected_count":sum(expected.values()),"actual_count":sum(actual.values())})
        config=yaml.safe_load((folder/"files/config.yaml").read_text())
        for name,value in src["config"].items():
            if name not in config or config[name]["value"]!=value:
                errors.append({"run":key,"config_mismatch":name,"expected":str(value)[:150],
                               "actual":str(config.get(name))[:150]})
        for f in run["artifacts"]:
            for field in ["preserved_path","swanlab_path"]:
                p=ROOT/f[field]
                if not p.is_file() or digest(p)!=f["sha256"]:
                    errors.append({"run":key,"artifact_mismatch":f[field]})
            artifacts+=1
        total+=sum(expected.values())
        results.append({"source":key,"native_id":exp["id"],"metric_points":sum(actual.values()),
                        "artifacts":len(run["artifacts"]),"all_scalar_values_and_steps_exact":not bool(missing or extra)})
    if len(experiments)!=len(sources):errors.append({"native_run_count":len(experiments),"source_count":len(sources)})
    if manifest.get("network_attempts"):errors.append({"migration_network_attempts":manifest["network_attempts"]})
    frozen=subprocess.check_output(["/snap/bin/uv","--no-config","pip","freeze","--python",str(ROOT/".venv/bin/python")],text=True)
    (M/"model_environment_after.txt").write_text(frozen)
    before=set((M/"model_environment_before.txt").read_text().splitlines())
    after=set(frozen.splitlines())
    if before-after:errors.append({"existing_packages_changed_or_removed":sorted(before-after)})
    report={"status":"verified" if not errors else "failed","runs":len(results),"metric_points":total,
            "artifact_count":artifacts,"max_scalar_difference":0 if not errors else None,
            "network_attempts_during_import":manifest.get("network_attempts"),
            "existing_model_packages_unchanged":not bool(before-after),
            "model_packages_added":len(after-before),
            "projects":[dict(r) for r in db.execute("select id,name,sum from project")],
            "per_run":results,"errors":errors,
            "no_training_or_scientific_evaluation":True,
            "timestamp_policy":"Original timestamps in source_export.json; native point creation time is import time."}
    (M/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="per_run"},ensure_ascii=False))
    if errors:raise SystemExit(1)
if __name__=="__main__":main()
