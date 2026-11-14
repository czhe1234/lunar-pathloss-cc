"""One-time server-local migration. Reads records only; never imports training code."""
import os,sys,json,hashlib,shutil,time,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/"tracking/swanlab";M=STATE/"migration"
os.environ.update(CUDA_VISIBLE_DEVICES="",SWANLAB_MODE="local",SWANLAB_ROOT=str(STATE/"root"),
 SWANLAB_LOGDIR=str(STATE/"logs"),SWANLAB_PROBE_MONITOR="false",SWANLAB_PROBE_SWANLAB="false",
 SWANLAB_API_HOST="http://127.0.0.1:16007",SWANLAB_LOG_LEVEL="error")
network_attempts=[]
def audit(event,args):
 if event=="socket.connect" and isinstance(args[1],tuple):
  network_attempts.append(str(args[1]));raise RuntimeError("Network transport is disabled during record migration")
sys.addaudithook(audit)
import swanlab
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 t=p.with_suffix(".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def main():
 bundle=json.loads((M/"source_export.json").read_text())
 names={str(e["experiment_id"]):e["name"] for e in bundle["experiments"]}
 records=[]
 for r in bundle["runs"]:
  rid=r["info"]["run_uuid"];exp=names[str(r["info"]["experiment_id"])]
  phase=r["tags"].get("cc.phase","")
  prefix="C146" if exp.startswith("C146") else "C147最终" if phase=="direct_covered" else "C147初始搜索"
  name=r["tags"].get("mlflow.runName") or r["info"].get("name") or rid
  records.append({"source_type":"mlflow","source_run_id":rid,"project":exp,"name":prefix+" | "+name,
    "config":r["params"],"source_info":r["info"],"source_tags":r["tags"],"metrics":r["metrics"],"artifacts":r["artifacts"]})
 c145=json.loads((M/"source_c145.json").read_text());records.append(c145)
 manifest_path=M/"manifest.json"
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"swanlab_version":swanlab.__version__,"mode":"local","runs":{}}
 settings=swanlab.Settings(probe={k:False for k in ["hardware","runtime","requirements","conda","git","swanlab","monitor"]})
 for src in records:
  key=src["source_type"]+":"+src["source_run_id"]
  if key in manifest["runs"]:
   assert manifest["runs"][key]["metric_points"]==len(src["metrics"]);continue
  rid="cc"+hashlib.sha256(key.encode()).hexdigest()[:19]
  config={**src["config"],"source_project":src["project"],"source_info":src.get("source_info",{}),"source_tags":src.get("source_tags",{}),
    "migration":{"source":src["source_type"],"source_run_id":src["source_run_id"],"import_only":True,
     "source_step_semantics":src.get("source_step_semantics","Same step as original MLflow series"),
     "original_timestamps":"tracking/swanlab/migration/source_export.json (for MLflow)",
     "source_summary":src.get("source_summary",{}),"artifact_count":len(src["artifacts"])}}
  description="历史记录迁移，没有重跑训练或实验。界面创建时间/时长为导入时间；原始时间与详细记录保存在source_info和源快照。"
  r=swanlab.init(project="Lunar_CC",name=src["name"],id=rid,resume="never",mode="local",log_dir=str(STATE/"logs"),
    config=config,description=description,settings=settings,tags=[src["project"].split("_")[0],"migrated"])
  folder=Path(r.dir);artifact_root=folder/"files/artifacts";artifact_root.mkdir(parents=True,exist_ok=True)
  copied=[]
  for f in src["artifacts"]:
   origin=ROOT/f["preserved_path"];assert sha(origin)==f["sha256"]
   target=artifact_root/f["path"];target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(origin,target);assert sha(target)==f["sha256"]
   copied.append({**f,"swanlab_path":str(target.relative_to(ROOT))})
  (folder/"files/source_artifacts.json").write_text(json.dumps(copied,ensure_ascii=False,indent=2)+"\n")
  by_step={}
  for x in src["metrics"]:
   row=by_step.setdefault(int(x["step"]),{})
   if x["key"] in row:assert row[x["key"]]==x["value"]
   row[x["key"]]=x["value"]
  for step,data in sorted(by_step.items()):swanlab.log(data,step=step)
  media_step=max(by_step,default=0)+1
  index="\n".join(f"- {f['path']} ({f['bytes']} bytes) — {f['sha256']}" for f in copied)
  swanlab.log({"migration/artifact_index":swanlab.Text(index or "No source artifacts")},step=media_step)
  report=next((ROOT/f["preserved_path"] for f in src["artifacts"] if f["path"].endswith(".md")),None)
  if report:swanlab.log({"migration/report":swanlab.Text(report.read_text())},step=media_step)
  swanlab.finish()
  manifest["runs"][key]={"source_type":src["source_type"],"source_run_id":src["source_run_id"],
    "project":src["project"],"name":src["name"],"swanlab_id":rid,"run_dir":str(folder.relative_to(ROOT)),
    "metric_points":len(src["metrics"]),"metric_keys":len({v["key"] for v in src["metrics"]}),"artifacts":copied}
  atomic(manifest_path,manifest)
  print("MIGRATED",key,len(src["metrics"]),len(copied),flush=True)
 manifest.update(status="import_complete",metric_points=sum(x["metric_points"] for x in manifest["runs"].values()),
  artifact_count=sum(len(x["artifacts"]) for x in manifest["runs"].values()),network_attempts=network_attempts)
 atomic(manifest_path,manifest)
 assert not network_attempts,network_attempts
 print("IMPORT_COMPLETE",len(manifest["runs"]),manifest["metric_points"],manifest["artifact_count"],flush=True)
if __name__=="__main__":main()
