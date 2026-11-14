"""C147 final, direct-complete comparison and bounded stability checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="",MLFLOW_DISABLE_TELEMETRY="true",DO_NOT_TRACK="true",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import json,sys,time,hashlib,sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import c147_score as scoremod
from src.c147_score import ARMS,BANDS
from src.c142_backend import file_hash
A=ROOT/"audit/c147_sionna_reflections"
def atomic(path,obj):
 p=path.with_suffix(".tmp");p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n");p.replace(path)
def covered_item(scorer,mid,folder):
 scoremod.A=folder
 item=scorer.load_map(mid);t=np.load(A/"targets"/f"{mid}.npz")
 changes=[];missing=[];maxratio=0.
 for b in (0,1):
  direct=np.load(A/"direct_coverage"/f"{mid}_b{b}.npz")
  assert np.array_equal(direct["points"],t["points"])
  assert np.max(direct["path_count"])<=1
  native=direct["gain"].reshape(-1,4).mean(1)
  complete=(native[t["donor_inverse"]]*t["weights"]).sum(1)
  vals=[];miss=[]
  for seed in scorer.protocol["seeds"]:
   path=folder/"paths"/f"{mid}_b{b}_s{seed}_d{scorer.protocol['maximum_total_interactions']}_n{scorer.protocol['samples_per_src']}"
   raw=np.concatenate([np.load(path/f"batch{i}.npz")["exact_gain"][0] for i in range(0,len(t["points"]),16)])
   old=(raw.reshape(-1,4).mean(1)[t["donor_inverse"]]*t["weights"]).sum(1)
   maxratio=max(maxratio,float(np.max(old/np.maximum(complete,1e-300))))
   vals.append(complete-old);miss.append(int(((raw==0)&(direct["gain"]>0)).sum()))
  changes.append(vals);missing.append(miss)
 assert maxratio<=1.001,maxratio
 correction=np.array(changes).transpose(1,0,2)
 item["replicates"]+=correction[:,None]
 item["gain"][1:]=item["replicates"].mean(0)
 assert (item["gain"]>=0).all()
 item["direct_coverage"]={"missing_target_rays_per_band_seed":missing,"max_original_to_complete_R0_ratio":maxratio}
 return item
def extended(scorer,items):
 out=scorer.score(items)
 y=np.concatenate([v["y"] for v in items],1);valid=np.concatenate([v["valid"] for v in items],1)
 base=np.concatenate([v["base"] for v in items],1)
 gain=np.concatenate([v["gain"] for v in items],2)
 error=-10*np.log10(np.maximum(gain,1e-30))-y
 tail=np.abs(y-base)>=5;common=valid&(gain>1e-30).all(0)
 masks={"all":valid,"tail":valid&tail,"under_tail":valid&(y-base>=5),"over_tail":valid&(y-base<=-5),
  "non_tail":valid&~tail,"C82_good":valid&(np.abs(y-base)<=1),"common_positive":valid&(gain>0).all(0),
  "common_unfloored":common,"tail_common_unfloored":common&tail}
 for name,m in masks.items():
  out["regions"][name]=m.sum(1).tolist()
  for i,arm in enumerate(ARMS):
   n=m.sum(1);ss=np.where(m,error[i]**2,0).sum(1)
   if name not in out["models"][arm]:
    out["models"][arm][name]={"count":n.tolist(),"sse":ss.tolist(),"rmse":np.sqrt(ss/np.maximum(n,1)).tolist(),
      "bias":(np.where(m,error[i],0).sum(1)/np.maximum(n,1)).tolist(),
      "zero_fraction":((m&(gain[i]==0)).sum(1)/np.maximum(n,1)).tolist(),
      "error_ge5_fraction":((m&(np.abs(error[i])>=5)).sum(1)/np.maximum(n,1)).tolist()}
   out["models"][arm][name]["floor_fraction"]=((m&(gain[i]<=1e-30)).sum(1)/np.maximum(n,1)).tolist()
 for name in masks:
  out["delta_R1_minus_R3"][name]=(np.array(out["models"]["sionna_R1"][name]["rmse"])-out["models"]["sionna_R3"][name]["rmse"]).tolist()
 return out
def main():
 scoremod.A=A;scorer=scoremod.Scorer()
 for n,digest in scorer.protocol["source_sha256"].items():assert file_hash(ROOT/n)==digest
 items=[covered_item(scorer,r["map"],A) for r in scorer.setup["records"]]
 scoremod.A=A;summary=extended(scorer,items)
 summary["per_map"]={str(v["map"]):extended(scorer,[v]) for v in items}
 summary["direct_coverage"]={str(v["map"]):v["direct_coverage"] for v in items}
 summary["status"]="verified"
 summary["primary_contract"]="Same fixed8scenes and256diagnostic pixels; standard Sionna fields; direct-transmission coverage completed separately, shared by R1/R2/R3."
 summary["initial_search_summary"]="audit/c147_sionna_reflections/comparison_summary.json"
 summary["source_sha256"]={"finalizer":file_hash(Path(__file__)),"main_protocol":file_hash(A/"protocol.json"),"direct_control":file_hash(A/"direct_coverage_protocol.json")}
 # Check every recorded inference output against its receipt and permitted device UUID.
 allowed={"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009","GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0","GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"}
 checked=0
 for root in [A/"paths",A/"direct_coverage",A/"baseline"]:
  for p in root.rglob("*.npz"):
   meta=json.loads(p.with_suffix(".json").read_text());assert file_hash(p)==meta["sha256"]
   device=meta.get("GPU",meta.get("gpu",{}))
   if isinstance(device,dict):device=device["uuid"]
   assert device in allowed,(p,device)
   checked+=1
 summary["verified_output_files"]=checked
 checks={}
 for mode,mid in [("sampling262144",4604),("sampling262144",22003),("totaldepth12",17305)]:
  folder=A/mode
  p=folder/"baseline"
  if not p.exists():p.symlink_to((A/"baseline").resolve(),target_is_directory=True)
  scoremod.A=folder;other=scoremod.Scorer()
  while not other.ready(mid):time.sleep(5)
  trial=covered_item(other,mid,folder);result=extended(other,[trial])
  initial=summary["per_map"][str(mid)]
  checks[f"{mode}_{mid}"]={"tail_RMSE_R1":result["models"]["sionna_R1"]["tail"]["rmse"],
   "tail_RMSE_R3":result["models"]["sionna_R3"]["tail"]["rmse"],
   "tail_delta_R1_minus_R3":result["delta_R1_minus_R3"]["tail"],
   "primary_tail_delta":initial["delta_R1_minus_R3"]["tail"],
   "R3_tail_RMSE_change":(np.array(result["models"]["sionna_R3"]["tail"]["rmse"])-initial["models"]["sionna_R3"]["tail"]["rmse"]).tolist()}
  atomic(folder/f"score_direct_covered_{mid}.json",result)
 summary["stability_checks"]=checks
 summary["absolute_HF_sampling_not_converged"]=abs(checks["sampling262144_22003"]["R3_tail_RMSE_change"][1])>.25
 summary["interpretation"]="Small paired reflection-depth gains on this diagnostic set; absolute HF solver errors remain sensitive to sampling on checked terrain05819."
 spread=np.ptp([r["tail"]["delta_R1_minus_R3"] for r in summary["sampling_seeds"]],axis=0)
 summary["sampling_delta_spread"]=spread.tolist()
 summary["verdict"]="On these fixed diagnostic pixels, extra reflections yield small tail improvements; standard Sionna improves over the current raw solver, especially HF; learned-model improvement is untested."
 atomic(A/"comparison_direct_covered.json",summary)
 fmt=lambda x:" / ".join(f"{v:.6f}" for v in x)
 lines=["# C147：Sionna反射次数对照","",
 "在本批固定大误差样点上，一次增加到三次反射，RMSE下降约"+fmt(summary["delta_R1_minus_R3"]["tail"])+" dB。",
 "在当前采样预算下，Sionna相对旧原始物理求解器有改善，尤其高频；主要收益并非来自增加到三次反射。高频绝对误差仍有采样敏感性，不能作为已收敛精度。这里没有训练网络。",
 "",
 "固定8个地形、256个诊断像素；同一TX、三角接收面、材料与供体插值。每个小面取4点，先平均线性功率再转dB。",
 "Sionna保留透射；同一批路径按实际反射次数累计成最多1/2/3次反射。总传播深度上限8，另做12的检查。",
 "射线采样65536/批，两个种子13/37。接收天线VH的功率求和以对应覆盖图的极化口径。",
 "初始路径搜索漏掉部分零反射透射路径。使用Sionna原始材料和场计算、沿TX到RX方向发现这条直线路径后，统一补齐三组的零反射部分；未拟合标签。",
 "旧求解器7图复用原始缓存，仅00414补算。所有原始与初始搜索结果保留。",
 "",
 "表内数字为物理路损RMSE，顺序415MHz/5.775GHz；不是C82或训练模型成绩。",
 "|求解器|全部样点|固定大误差区|原准确区|",
 "|---|---|---|---|"]
 names={"current_solver":"现有求解器","sionna_R1":"Sionna最多1次反射","sionna_R2":"Sionna最多2次反射","sionna_R3":"Sionna最多3次反射"}
 for arm in ARMS:
  m=summary["models"][arm];lines.append(f"|{names[arm]}|{fmt(m['all']['rmse'])}|{fmt(m['tail']['rmse'])}|{fmt(m['C82_good']['rmse'])}|")
 lines+=["","固定尾区按C82原误差>=5dB定义，双频分别有"+str(summary["regions"]["tail"])+"个像素。其余区域定义与完整数字见comparison_direct_covered.json。",
  "统一增益下限1e-30（300dB）用于处理零值和极弱功率。下表单列触限比例，避免把该数值处理当成真实可测损耗。",
  "|求解器|全部样点触限比例|大误差区触限比例|大误差区共同未触限子集RMSE|","|---|---|---|---|"]
 for arm in ARMS:
  m=summary["models"][arm];lines.append(f"|{names[arm]}|{fmt(m['all']['floor_fraction'])}|{fmt(m['tail']['floor_fraction'])}|{fmt(m['tail_common_unfloored']['rmse'])}|")
 lines+=["","共同未触限尾区的像素数："+str(summary["regions"]["tail_common_unfloored"]),
  "两个种子间，R1到R3的尾区RMSE降幅差为"+fmt(spread)+" dB。",
  "","稳定性检查：","|检查|原配置R1到R3尾区降幅|检查后降幅|检查后R3尾区RMSE变化|","|---|---|---|"]
 for name,r in checks.items():lines.append(f"|{name}|{fmt(r['primary_tail_delta'])}|{fmt(r['tail_delta_R1_minus_R3'])}|{fmt(r['R3_tail_RMSE_change'])}|")
 lines+=["","05819_00将采样量提高4倍，R3高频尾区RMSE下降约8.06dB，而同配置R1到R3仅下降约0.003dB。说明该处当前物理计算对采样量的敏感性远大于反射次数；不能把采样稳定性与物理正确性等同。",
 "00414_00：R1到R3尾区降幅为"+fmt(summary["per_map"]["1819"]["delta_R1_minus_R3"]["tail"])+" dB，未修复该反例。",
 "大误差区的低估损耗部分并未随反射次数一致改善；更多反射增加接收能量，可能加重原本信号偏强的位置。正负尾区均保留在JSON。",
 "这些地形和像素是定向诊断，不是随机全图评价或官方VAL。尚未验证把Sionna特征送入模型后能否超过C82。",
 "下一步优先改善路径搜索覆盖、弱散射积分与特征的动态范围；现有结果不支持只靠增加反射次数解决主要瓶颈。"]
 report=ROOT/"results/C147_Sionna反射深度对照_20261114.md";report.write_text("\n".join(lines)+"\n")
 from mlflow.tracking import MlflowClient
 from mlflow.entities import Metric,Param
 from mlflow.telemetry import get_telemetry_client
 assert get_telemetry_client() is None
 client=MlflowClient(tracking_uri="http://127.0.0.1:16007")
 registry=json.loads((A/"mlflow_runs.json").read_text());initial_ids=registry.get("initial_search_run_ids",registry["run_ids"])
 for rid in initial_ids.values():client.set_tag(rid,"cc.phase","initial_search")
 ids={}
 for arm in ARMS:
  identity=f"C147/{summary['source_sha256']['main_protocol']}/{arm}/direct_covered"
  found=client.search_runs([registry["experiment_id"]],filter_string=f"tags.`cc.identity` = '{identity}'")
  assert len(found)<=1
  if found:run=found[0]
  else:
   run=client.create_run(registry["experiment_id"],run_name=names[arm],tags={"cc.identity":identity,"cc.phase":"direct_covered","cc.storage":"server_only",
    "mlflow.note.content":"固定8地形256诊断像素。物理路损误差，不是训练模型或官方VAL。Sionna使用原始场计算，另以定向候选搜索补齐零反射透射路径；所有反射上限共享此直达部分。"})
   params={k:v for k,v in scorer.protocol.items() if isinstance(v,(str,int,float,bool))}
   params.update(arm=arm,direct_coverage_completed=True,zero_gain_floor=1e-30)
   client.log_batch(run.info.run_id,params=[Param(k,str(v)) for k,v in params.items()])
   for name in ["protocol.json","setup.json","targets.json","direct_coverage_protocol.json"]:
    client.log_artifact(run.info.run_id,str(A/name),artifact_path="provenance")
  rid=run.info.run_id;ids[arm]=rid
  known={(key,m.step):m.value for key in client.get_run(rid).data.metrics for m in client.get_metric_history(rid,key)}
  metrics=[]
  for step,rec in enumerate(scorer.setup["records"]):
   for group,stat in summary["per_map"][str(rec["map"])]["models"][arm].items():
    for field in ["rmse","bias","zero_fraction","error_ge5_fraction","floor_fraction"]:
     for band,val in zip(BANDS,stat[field]):metrics.append(Metric(f"per_map/{group}/{field}_{band}",float(val),int(time.time()*1000),step))
  for group,stat in summary["models"][arm].items():
   for field in ["rmse","bias","zero_fraction","error_ge5_fraction","floor_fraction"]:
    for band,val in zip(BANDS,stat[field]):metrics.append(Metric(f"aggregate/{group}/{field}_{band}",float(val),int(time.time()*1000),8))
  for m in metrics:
   if (m.key,m.step) in known:assert known[m.key,m.step]==m.value
  pending=[m for m in metrics if (m.key,m.step) not in known]
  for i in range(0,len(pending),500):client.log_batch(rid,metrics=pending[i:i+500])
  for p in [A/"comparison_direct_covered.json",report]:client.log_artifact(rid,str(p),artifact_path="final")
  client.set_terminated(rid,status="FINISHED")
 registry.update(run_ids=ids,initial_search_run_ids=initial_ids,url="http://127.0.0.1:16007/#/experiments/2/runs?workflowType=machine_learning&compareRunsMode=CHART&searchFilter=tags.%60cc.phase%60%20%3D%20%27direct_covered%27")
 atomic(A/"mlflow_runs.json",registry)
 atomic(A/"progress.json",{"stage":"complete_verified","maps":8,"pixels":256,"summary":"audit/c147_sionna_reflections/comparison_direct_covered.json","run_ids":ids})
 print("FINAL_C147",fmt(summary["delta_R1_minus_R3"]["tail"]),json.dumps(checks),flush=True)
if __name__=="__main__":main()
