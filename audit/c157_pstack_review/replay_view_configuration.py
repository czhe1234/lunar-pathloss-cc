"""C157 minimal in-memory replay of the SwanLab view configuration function."""
from pathlib import Path
import ast,hashlib,json,sqlite3,types,datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"scripts/configure_swanlab_view.py"
class MemoryPath:
 def __init__(self,key,store):self.key=key;self.store=store
 def __truediv__(self,other):return MemoryPath(self.key+"/"+str(other),self.store)
 def exists(self):return self.key in self.store
 def read_text(self):return self.store[self.key]
 def write_text(self,text):self.store[self.key]=text;return len(text)
 def __str__(self):return self.key
class Connection:
 def __init__(self,conn):self.conn=conn
 @property
 def row_factory(self):return self.conn.row_factory
 @row_factory.setter
 def row_factory(self,value):self.conn.row_factory=value
 def __getattr__(self,name):return getattr(self.conn,name)
 def close(self):pass
def run_case(add_metric):
 original=sqlite3.connect("file:"+str(ROOT/"tracking/swanlab/logs/runs.swanlab")+"?mode=ro",uri=True)
 database=sqlite3.connect(":memory:");original.backup(database);original.close();database.row_factory=sqlite3.Row
 before=json.loads((ROOT/"tracking/swanlab/view/paired_rmse_before_32.json").read_text())
 store={"view/paired_rmse_before_32.json":json.dumps(before)}
 if add_metric:
  row=dict(database.execute("select * from tag where experiment_id=32 limit 1").fetchone());row.pop("id")
  row.update(name="new/legitimate_metric",folder="new_metric_folder",sort=99999)
  database.execute("insert into tag("+",".join(row)+") values("+",".join("?" for _ in row)+")",tuple(row.values()));database.commit()
 initial={x["id"]:x["status"] for x in database.execute("select id,status from chart where experiment_id=32")}
 tree=ast.parse(SOURCE.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="configure_paired_rmse")
 globals_={"sqlite3":types.SimpleNamespace(connect=lambda *args,**kwargs:Connection(database),Row=sqlite3.Row),"json":json,
  "VIEW":MemoryPath("view",store),"STATE":MemoryPath("state",store),"ROOT":ROOT}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),"exec"),globals_)
 caught=None;returned=None
 try:returned=globals_["configure_paired_rmse"](32,{"baseline":{"monitor":[2.1463905073492664,2.418014484925531]}})
 except BaseException as error:caught={"type":type(error).__name__,"message":str(error)}
 final={x["id"]:x["status"] for x in database.execute("select id,status from chart where experiment_id=32")}
 changed=[{"chart_id":k,"before":v,"after":final[k]} for k,v in initial.items() if final[k]!=v]
 database.close()
 return {"added_legitimate_metric":add_metric,"error":caught,"view_changes_committed":changed,"returned_success":returned is not None}
if __name__=="__main__":
 control=run_case(False);new_metric=run_case(True)
 # Failure signal: nonzero exit if the clean control fails or the expected stale-backup trigger is absent.
 assert control["error"] is None and control["returned_success"]
 assert new_metric["error"]=={"type":"AssertionError","message":"Raw tag metadata changed"}
 assert new_metric["view_changes_committed"]
 result={"status":"reproduced","source":"scripts/configure_swanlab_view.py","source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"Original function AST, read-only source database copied into SQLite :memory:, all view file IO virtualized in memory",
  "control":control,"legitimate_new_metric":new_metric,"original_database_modified":False,"source_modified":False,"training":False,"GPU_used":False}
 (OUT/"view_configuration_replay.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(result,ensure_ascii=False))
