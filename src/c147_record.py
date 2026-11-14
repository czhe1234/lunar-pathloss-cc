"""Server-only MLflow records for C147; source JSON and arrays remain authoritative."""
import os
os.environ.update(MLFLOW_DISABLE_TELEMETRY="true",DO_NOT_TRACK="true",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import sys,json,time,hashlib,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.c147_score import Scorer,ARMS,BANDS
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric,Param
from mlflow.telemetry import get_telemetry_client
A=ROOT/"audit/c147_sionna_reflections"
def atomic(path,obj):
 p=path.with_suffix(".tmp");p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n");p.replace(path)
def main():
 assert get_telemetry_client() is None
 protocol=json.loads((A/"protocol.json").read_text());digest=hashlib.sha256((A/"protocol.json").read_bytes()).hexdigest()
 client=MlflowClient(tracking_uri="http://127.0.0.1:16007")
 exp=client.get_experiment_by_name("C147_sionna_reflections")
 eid=exp.experiment_id if exp else client.create_experiment("C147_sionna_reflections")
 ids={};known={}
 for arm in ARMS:
  identity=f"C147/{digest}/{arm}"
  found=client.search_runs([eid],filter_string=f"tags.`cc.identity` = '{identity}'")
  assert len(found)<=1
  if found:run=found[0]
  else:
   run=client.create_run(eid,run_name=arm,tags={"cc.identity":identity,"cc.storage":"server_only",
    "mlflow.note.content":"8地形256个固定诊断像素；横轴为地形序号。不是训练曲线或官方VAL。Sionna按同一批路径中反射次数累计，透射保持开启。"})
   params={k:v for k,v in protocol.items() if isinstance(v,(str,int,float,bool))}
   params.update(arm=arm,reflection_cap=1 if arm=="current_solver" else int(arm[-1]),seeds=str(protocol["seeds"]))
   client.log_batch(run.info.run_id,params=[Param(k,str(v)) for k,v in params.items()])
   for name in ["protocol.json","setup.json","targets.json"]:
    client.log_artifact(run.info.run_id,str(A/name),artifact_path="provenance")
  ids[arm]=run.info.run_id
  known[arm]={(key,m.step):m.value for key in client.get_run(run.info.run_id).data.metrics
    for m in client.get_metric_history(run.info.run_id,key)}
 atomic(A/"mlflow_runs.json",{"experiment_id":eid,"run_ids":ids,"url":f"http://127.0.0.1:16007/#/experiments/{eid}/runs?workflowType=machine_learning&compareRunsMode=CHART"})
 scorer=Scorer();done={}
 while True:
  for i,rec in enumerate(scorer.setup["records"]):
   mid=rec["map"]
   if mid in done or not scorer.ready(mid):continue
   item=scorer.load_map(mid);done[mid]=item;summary=scorer.score([item])
   atomic(A/f"score_{mid}.json",summary)
   for arm in ARMS:
    metrics=[]
    for group,stat in summary["models"][arm].items():
     for field in ["rmse","bias","zero_fraction","error_ge5_fraction"]:
      for band,val in zip(BANDS,stat[field]):
       metrics.append(Metric(f"per_map/{group}/{field}_{band}",float(val),int(time.time()*1000),i))
    pending=[m for m in metrics if (m.key,m.step) not in known[arm]]
    for m in metrics:
     if (m.key,m.step) in known[arm]:assert known[arm][m.key,m.step]==m.value
    if pending:
     client.log_batch(ids[arm],metrics=pending)
     known[arm].update({(m.key,m.step):m.value for m in pending})
    client.set_tag(ids[arm],"cc.completed_maps",str(len(done)))
   print("SCORED_MAP",mid,len(done),flush=True)
  progress={"updated":datetime.now().astimezone().isoformat(),"completed_maps":list(done),"expected_maps":[r["map"] for r in scorer.setup["records"]],"pid":os.getpid(),"run_ids":ids}
  atomic(A/"progress.json",progress)
  if len(done)==len(scorer.setup["records"]):break
  time.sleep(5)
 items=[done[r["map"]] for r in scorer.setup["records"]]
 summary=scorer.score(items)
 summary["protocol_sha256"]=digest
 summary["completed"]=datetime.now().astimezone().isoformat()
 delta=summary["delta_R1_minus_R3"]["tail"]
 stable=[s["tail"]["delta_R1_minus_R3"] for s in summary["sampling_seeds"]]
 summary["sampling_delta_spread"]=(np.max(stable,0)-np.min(stable,0)).tolist()
 summary["sampling_followup_needed"]=any(v>.1 for v in summary["sampling_delta_spread"])
 summary["verdict"]="R3 improves fixed-tail RMSE in both bands" if min(delta)>=.05 and all(min(v)>0 for v in stable) else "No material dual-band fixed-tail improvement from R1 to R3"
 atomic(A/"comparison_summary.json",summary)
 lines=["# C147：Sionna 反射次数对照","",summary["verdict"],"",
 "固定8地形、256个诊断像素。数字顺序415MHz/5.775GHz。这里没有网络训练、标签校准或官方VAL。",
 "同一三角地表、TX、地表上方1米接收面、材料。每个接收小面取4点，使用同一供体与线性功率插值。",
 "Sionna PathSolver保留透射，统一最大8次传播交互，再按实际反射次数累计成R<=1/2/3。两个种子13/37，65536条采样射线/批。",
 "旧求解器沿用一次反射、128方向散射的原配置；7图复用旧缓存，仅00414补算。旧积分近似与Sionna有所不同。",
 "","|方案|全部样点RMSE(dB)|固定大误差区RMSE(dB)|大误差区偏差(dB)|零功率比例|","|---|---|---|---|---|"]
 fmt=lambda x:" / ".join(f"{v:.6f}" for v in x)
 for arm in ARMS:
  m=summary["models"][arm]
  lines.append(f"|{arm}|{fmt(m['all']['rmse'])}|{fmt(m['tail']['rmse'])}|{fmt(m['tail']['bias'])}|{fmt(m['all']['zero_fraction'])}|")
 lines+=["","固定尾区按C82原误差>=5dB定义。样本是定向诊断，不能外推为全图或总体成绩。",
 "零功率评分统一按增益1e-30计，同时另报共同正功率子集；完整区域和逐图数字见comparison_summary.json与score_*.json。",
 "R1到R3尾区RMSE降幅："+fmt(delta)+" dB；两个采样种子间该降幅的差："+fmt(summary["sampling_delta_spread"])+" dB。",
 "最大直达自由空间功率相对差："+str(summary["max_direct_FSPL_relative_error"]),
 "最大传播交互深度边界路径占R3功率比例："+fmt(summary["maximum_total_depth_ceiling_power_fraction"])]
 report=ROOT/"results/C147_Sionna反射深度对照_20261114.md";report.write_text("\n".join(lines)+"\n")
 for arm in ARMS:
  metrics=[Metric(f"aggregate/{group}/{field}_{band}",float(val),int(time.time()*1000),len(items))
   for group,stat in summary["models"][arm].items() for field in ["rmse","bias","zero_fraction","error_ge5_fraction"] for band,val in zip(BANDS,stat[field])]
  pending=[m for m in metrics if (m.key,m.step) not in known[arm]]
  if pending:client.log_batch(ids[arm],metrics=pending)
  for p in [A/"comparison_summary.json",report]:client.log_artifact(ids[arm],str(p),artifact_path="final")
  client.set_terminated(ids[arm],status="FINISHED")
 atomic(A/"progress.json",{**progress,"stage":"complete","summary":"audit/c147_sionna_reflections/comparison_summary.json","run_ids":ids})
 print("C147_COMPLETE",summary["verdict"],flush=True)
if __name__=="__main__":main()
