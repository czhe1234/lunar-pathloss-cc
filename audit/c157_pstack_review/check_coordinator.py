"""C157 read-only coordinator checks; no model imports or scientific experiment reruns."""
from pathlib import Path
import ast,hashlib,json,sqlite3,urllib.request,datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def run():
 plan=read(OUT/"protocol.json")
 python_files=[ROOT/p for p in plan["source_bindings"] if Path(p).suffix==".py"]
 if not python_files:raise RuntimeError("Syntax scan matched zero files")
 syntax_errors=[]
 for file in python_files:
  try:ast.parse(file.read_text(),filename=str(file))
  except SyntaxError as error:syntax_errors.append({"file":str(file.relative_to(ROOT)),"line":error.lineno,"message":error.msg})
 # Failure signal: a nonempty syntax_errors list, or zero Python files.
 with urllib.request.urlopen("http://127.0.0.1:16007/cc/data.json",timeout=20) as response:
  if response.status!=200:raise RuntimeError("Overview HTTP status is not 200")
  dashboard=json.load(response)
 if not dashboard.get("profiles"):raise RuntimeError("Overview returned zero profiles")
 curve_checks=[];points=0
 for profile in dashboard["profiles"]:
  keys=[k for k in profile["sources"] if k.endswith("_history")]
  if not keys or not profile["id"].startswith("native_"):continue
  history=read(ROOT/dashboard["sources"][keys[0]]["path"])
  for panel in ("fit","monitor"):
   expected=[[point["epoch"],*point["evaluation"][panel]["rmse"]] for point in history]
   if profile["series"][panel]!=expected:raise RuntimeError(f"Overview differs from history: {profile['id']}/{panel}")
   points+=len(expected)*2
  curve_checks.append({"profile":profile["id"],"points_per_band":len(history),"both_panels_exact":True})
 # Failure signal: nonzero exit on any value/step mismatch, or zero checked histories.
 if not curve_checks:raise RuntimeError("No native histories were verified")
 state=ROOT/"tracking/swanlab";db=sqlite3.connect("file:"+str(state/"logs/runs.swanlab")+"?mode=ro",uri=True)
 db.row_factory=sqlite3.Row
 raw_checks=[]
 for experiment_id,folder in [(29,"runs/c153_standalone_c151a"),(32,"runs/c155_unet")]:
  exp=dict(db.execute("select * from experiment where id=?",(experiment_id,)).fetchone())
  history=read(ROOT/folder/"history.json")
  tags=[dict(x) for x in db.execute("select * from tag where experiment_id=?",(experiment_id,))]
  checked=0
  for panel in ("fit","monitor"):
   for band_index,band in enumerate(("415MHz","5p775GHz")):
    expected={row["epoch"]:row["evaluation"][panel]["rmse"][band_index] for row in history}
    tag=next(x for x in tags if x["name"]==f"{panel}/RMSE_{band}_dB")
    files=list((state/"logs"/exp["run_id"]/"logs"/tag["folder"]).glob("*.log"))
    if not files:raise RuntimeError("Raw SwanLab tag matched zero log files")
    points_actual=[json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    actual={x["index"]:x["data"] for x in points_actual}
    if actual!=expected or len(points_actual)!=len(expected):raise RuntimeError("Raw SwanLab RMSE points differ from history")
    checked+=len(expected)
  raw_checks.append({"experiment_id":experiment_id,"run_id":exp["run_id"],"RMSE_curves":4,"raw_points_verified":checked})
 # Failure signal: nonzero exit on missing tags, zero files, duplicate/missing points, or changed values.
 backups=[]
 for file in sorted((state/"view").glob("paired_*_before_*.json")):
  before=read(file);experiment_id=before["experiment"]["id"]
  current=[dict(x) for x in db.execute("select * from tag where experiment_id=?",(experiment_id,))]
  backups.append({"file":str(file.relative_to(ROOT)),"experiment_id":experiment_id,"tags_before":len(before["tag_rows"]),"tags_now":len(current),"unchanged":before["tag_rows"]==current})
 if not backups:raise RuntimeError("Found zero paired-view backups")
 # Failure signal: any backup entry has unchanged=false, or no backups.
 release=ROOT/"deliveries/github_lunar_pathloss_cc_20261116"
 manifest=read(release/"repository/SOURCE_SNAPSHOT.json");inventory=read(release/"inventory.json");verification=read(release/"upload_verification.json")
 if verification["commit"]!=plan["source_publication_commit"]:raise RuntimeError("Publication commit binding differs")
 if sha(release/"repository/SOURCE_SNAPSHOT.json")!=inventory["source_snapshot_sha256"]:raise RuntimeError("Public manifest SHA differs")
 for item in inventory["files_manifest"]:
  if sha(release/"repository"/item["path"])!=item["sha256"]:raise RuntimeError("Exported file SHA differs: "+item["path"])
 source_changed=[]
 for name,expected in plan["source_bindings"].items():
  row=next(x for x in manifest["source_files"] if x["path"]==name)
  file=ROOT/row.get("source_original_path",name)
  expected_sha=expected["sha256"]
  if name=="HANDOFF.md" and (OUT/"review.json").exists():
   expected_sha=read(OUT/"review.json").get("handoff_review_sha256",expected_sha)
  if sha(file)!=expected_sha:source_changed.append(name)
 # Failure signal: nonzero exit on any manifest/export mismatch; nonempty source_changed.
 result={"status":"coordinator_checks_complete","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "protocol_sha256":sha(OUT/"protocol.json"),"script_sha256":sha(Path(__file__)),
  "syntax":{"files":len(python_files),"errors":syntax_errors},"overview":{"HTTP":200,"profiles":len(dashboard["profiles"]),"default_profile":dashboard["default_profile"],"history_checks":curve_checks,"values_verified":points},
  "SwanLab_raw_RMSE":raw_checks,"paired_view_backups":backups,
  "publication":{"commit":verification["commit"],"tree":verification["tree"],"visibility":verification["visibility"],"exported_files_SHA_verified":len(inventory["files_manifest"])},
  "source_changed_during_review":source_changed,"training":False,"neural_inference":False,"GPU_used":False}
 (OUT/"coordinator_checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({"syntax_files":len(python_files),"syntax_errors":syntax_errors,"overview_histories":len(curve_checks),"overview_values_verified":points,"raw_SwanLab_points":sum(x["raw_points_verified"] for x in raw_checks),
  "backup_changes":[x for x in backups if not x["unchanged"]],"export_files_verified":len(inventory["files_manifest"]),"source_changes":source_changed},ensure_ascii=False))
 if syntax_errors or source_changed or any(not x["unchanged"] for x in backups):raise SystemExit(1)
if __name__=="__main__":run()
