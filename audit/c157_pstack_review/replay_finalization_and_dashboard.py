"""C157 CPU AST checks for terminal recovery and the maintained dashboard verifier."""
from pathlib import Path
import ast,hashlib,json,math,urllib.request,datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def resume_case(relative):
 file=ROOT/relative;tree=ast.parse(file.read_text());parents={}
 for parent in ast.walk(tree):
  for child in ast.iter_child_nodes(parent):parents[child]=parent
 loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter).replace(" ","")=="range(last_epoch+1,EPOCHS+1)")
 fn=loop
 while not isinstance(fn,ast.FunctionDef):fn=parents[fn]
 stores=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Name) and n.id=="rate" and isinstance(n.ctx,ast.Store)]
 if not stores or any(not loop.lineno<=line<=loop.end_lineno for line in stores):raise RuntimeError("The extracted recovery boundary omits another rate binding")
 check=next(n for n in ast.walk(fn) if isinstance(n,ast.Assert) and n.lineno>loop.end_lineno and "isclose(rate" in ast.unparse(n.test).replace(" ",""))
 probe=ast.parse("def boundary(last_epoch,updates):\n pass\n").body[0];probe.body=[loop,check]
 module=ast.fix_missing_locations(ast.Module(body=[probe],type_ignores=[]))
 env={"EPOCHS":60,"UPDATES":21120,"math":math};exec(compile(module,str(file),"exec"),env)
 caught=None
 try:env["boundary"](60,21120)
 except BaseException as error:caught={"type":type(error).__name__,"message":str(error)}
 # Failure signal: nonzero exit if original AST at the empty-loop boundary does not reproduce UnboundLocalError.
 assert caught and caught["type"]=="UnboundLocalError",caught
 return {"source":relative,"source_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"function":fn.name,
  "loop_lines":[loop.lineno,loop.end_lineno],"rate_store_lines":stores,"final_assert_line":check.lineno,"last_epoch":60,"updates":21120,
  "observed":caught,"scope":"Original loop/assert AST; all original rate stores proved inside skipped loop; no model or checkpoint imported"}
def dashboard_case():
 source=ROOT/"scripts/verify_swanlab_dashboard.py";tree=ast.parse(source.read_text())
 with urllib.request.urlopen("http://127.0.0.1:16007/cc/data.json",timeout=20) as response:
  assert response.status==200
  actual=json.load(response)
 checks=[]
 for node in ast.walk(tree):
  if not isinstance(node,ast.Assert):continue
  expression=ast.unparse(node.test)
  if "x['comparison']" not in expression and "x['records']" not in expression:continue
  value=eval(compile(ast.Expression(node.test),str(source),"eval"),{"x":actual,"len":len})
  checks.append({"line":node.lineno,"original_assertion":expression,"passes_current_records":bool(value)})
 # Failure signal: nonzero exit if no original assertions were evaluated or the documented stale counts are not reproduced.
 assert checks
 failures=[x for x in checks if not x["passes_current_records"]]
 assert any("records" in x["original_assertion"] for x in failures)
 assert any("monitor" in x["original_assertion"] for x in failures)
 return {"source":"scripts/verify_swanlab_dashboard.py","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
  "actual_records":actual["records"],"actual_comparison_rows":{k:len(v["rows"]) for k,v in actual["comparison"].items()},
  "original_assertions":checks,"scope":"Original assertion expressions evaluated against current server response; legacy script itself not run or allowed to write"}
if __name__=="__main__":
 result={"status":"reproduced","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "terminal_resume":[resume_case("src/c153_train.py"),resume_case("src/c155_train.py")],"legacy_dashboard_verifier":dashboard_case(),
  "training":False,"GPU_used":False,"neural_inference":False,"original_code_weights_and_records_modified":False}
 (OUT/"boundary_replay.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(result,ensure_ascii=False))
