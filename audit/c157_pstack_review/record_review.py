"""Record the completed C157 source review in server-local SwanLab."""
from pathlib import Path
import hashlib,json,os,sqlite3,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def main():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU-only audit")
 if (OUT/"tracking.json").exists():raise RuntimeError("C157 tracking already exists; do not duplicate the review")
 review=read(OUT/"review.json")
 if review["report_independent_review"]["status"]!="pass":raise RuntimeError("Review report has not passed independent document review")
 from src import cc_tracking
 import swanlab
 run=cc_tracking.init(name="C157 pstack project review",campaign="C157",
  config={"run_kind":"analysis.project_review","_cc_dashboard":{"kind":"analysis"},"verdict":review["verdict"],
   "source_publication_commit":review["source_publication_commit"],"protocol_sha256":sha(OUT/"protocol.json"),
   "full_read_files":review["full_read_unique_files"],"source_binding_count":len(read(OUT/"protocol.json")["source_bindings"]),
   "findings":{"P1":review["P1"],"P2":review["P2"],"P3":review["P3"]},
   "review_report":"audit/c157_pstack_review/report.md","report_sha256":sha(OUT/"report.md"),
   "documentation_review":"pass","training":False,"neural_inference":False,"GPU_used":False,
   "original_scientific_code_modified":False,"model_weights_modified":False,
   "reviewer_models":"inherit-parent; actual model/effort not returned; same-model possible"},
  description="只读源码与既有产物审查、CPU边界复现；报告及原记录留在CC。",tags=["pstack","review","CPU"])
 tracking={"run_id":run.id,"dir":str(run.dir),"mode":"local","project":"Lunar_CC"}
 write(OUT/"tracking.json",tracking)
 try:cc_tracking.log({"review/report":swanlab.Text((OUT/"report.md").read_text(),caption="C157 pstack 全项目审查")},step=1)
 finally:cc_tracking.finish()
 database=ROOT/"tracking/swanlab/logs/runs.swanlab"
 record=None;tags=[]
 for _ in range(10):
  with sqlite3.connect("file:"+str(database)+"?mode=ro",uri=True) as db:
   db.row_factory=sqlite3.Row
   row=db.execute("select * from experiment where run_id=?",(Path(tracking["dir"]).name,)).fetchone()
   if row:
    record=dict(row);tags=[dict(x) for x in db.execute("select * from tag where experiment_id=?",(record["id"],))]
  if record and any(x["name"]=="review/report" for x in tags):break
  time.sleep(.5)
 # Failure signal: missing experiment or report tag after ingestion, not merely HTTP200 from the SPA.
 if not record or not any(x["name"]=="review/report" for x in tags):raise RuntimeError("Local SwanLab did not ingest the review report")
 tracking.update(native_experiment_id=record["id"],view_url=f"http://127.0.0.1:16007/experiment/{record['id']}/chart",report_tag_verified=True)
 write(OUT/"tracking.json",tracking)
 print(json.dumps(tracking,ensure_ascii=False))
if __name__=="__main__":main()
