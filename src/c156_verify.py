"""C156 independent CPU verification and presentation finalization; no inference."""
from pathlib import Path
import hashlib,json,os,sqlite3,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"audit/c156_attribution"
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def ident(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def close(a,b,atol=1e-9):np.testing.assert_allclose(a,b,rtol=1e-10,atol=atol)
def run():
 assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
 d=read(OUT/"result.json");p=read(OUT/"protocol.json")
 assert d["protocol_sha256"]==sha(OUT/"protocol.json")
 assert d["training"] is False and d["neural_inference"] is False and d["CUDA_initialized"] is False
 assert d["scope"]["monitor_maps"]==628 and d["scope"]["terrains"]==128
 assert all(ident(ROOT/f)==v for f,v in p["input_file_identities"].items())
 revisions=[]
 existing=read(OUT/"verification.json") if (OUT/"verification.json").exists() else {}
 for f,h in p["source_sha256"].items():
  if f=="src/c156_figures.py" and existing.get("presentation_revisions"):
   assert sha(ROOT/f)==existing["presentation_revisions"][0]["new_sha256"]
  else:assert sha(ROOT/f)==h,(f,sha(ROOT/f),h)
 parent=read(ROOT/"audit/c155_unet/comparison_summary.json")
 assert sha(ROOT/"audit/c155_unet/comparison_summary.json")==p["parent_verified_sha256"]
 assert parent["selected_epoch"]==52
 for key,target in [("all","all"),("C82_good","C82_good"),("C82_tail","tail")]:
  old=parent["selected"]["monitor"]["groups"][target]
  for stage,k in [("C155","RMSE"),("C82","C82_RMSE")]:
   close(d["scores"][key][stage]["RMSE_dB"],old[k])
  close(d["scores"][key]["C155"]["count"],old["count"],atol=0)
  close(d["scores"][key]["C155"]["SSE_dB2"],old["SSE"],atol=1e-5)
 verified=read(ROOT/"runs/c155_unet/selected_verified.json")
 assert sha(ROOT/"runs/c155_unet/selected_monitor_prediction.npy")==verified["panels"]["monitor"]["prediction_sha256"]
 assert sha(ROOT/"runs/c155_unet/selected.pt")==read(ROOT/"runs/c155_unet/selected_export.json")["checkpoint_sha256"]
 parent_plan=read(ROOT/"audit/c155_unet/protocol.json")
 assert all(ident(ROOT/f)==v for f,v in parent_plan["protected_file_identities"].items())
 assert sha(ROOT/"audit/c88_mesh/base.npy")==parent_plan["protected_readonly_sha256"]["audit/c88_mesh/base.npy"]
 for scope,tab in d["scores"].items():
  for stage,x in tab.items():
   close(np.sqrt(np.asarray(x["SSE_dB2"])/x["count"]),x["RMSE_dB"])
 common=d["physical"]["common_tail"]
 close(np.sum(common["class_pixel_count"],axis=0),common["count"],atol=0)
 close(np.sum(common["class_percent"],axis=0),[100,100])
 for k in ["common_Clear","common_Grazing"]:
  close(d["physical"][k]["crossing_count_n_mean"],[0,0],atol=0)
 close(common["crossing_count_n_mean"],2*np.asarray(common["blocked_segments_mean"]))
 close(np.sum(common["blocked_segment_percent"],axis=0),[100,100])
 for tab in d["C155_excess_MSE_attribution"].values():assert tab["sum_matches_actual_gain"]
 sh=d["C82_good_diagnostics"]["integer_shift"]
 for b in (0,1):
  grid=np.array(sh["SSE_grid"])[:,:,b];assert np.unravel_index(grid.argmin(),grid.shape)==(2,2)
  close(sh["identity_RMSE_dB"][b],np.sqrt(grid[2,2]/sh["count"][b]))
 for sup in d["C82_good_diagnostics"]["roughness"].values():
  for x in sup.values():
   close(np.square(x["highpass_error_RMS_dB"])+np.square(x["localmean_error_RMS_dB"])+x["cross_term_dB2"],x["error_MSE_dB2"])
 stages={}
 for scope in ("all","C82_good"):
  t=d["scores"][scope];m={k:np.square(v["RMSE_dB"]) for k,v in t.items()}
  stages[scope]={}
  for before,after in [("base","local"),("base","graph"),("graph","blend"),("blend","calibration_replay")]:
   stages[scope][before+" -> "+after]={"MSE_reduction_dB2":(m[before]-m[after]).tolist(),"MSE_reduction_percent":(100*(1-m[after]/m[before])).tolist()}
  average=.5*(m["local"]+m["graph"])
  stages[scope]["mean_of_expert_MSE -> fused_prediction_MSE"]={"MSE_reduction_dB2":(average-m["blend"]).tolist(),"MSE_reduction_percent":(100*(1-m["blend"]/average)).tolist(),"meaning":"parallel expert complementarity; not serial gains to add twice"}
 z=np.load(OUT/"crossing_counts_sparse.npz")
 assert int(z["offsets"][-1])==d["CPU_geometry_rays"]==len(z["pixels"])==2015853
 cases_verified=[]
 rows=None;gt=None;masks=None;pred=None;c82=None;cidx=None
 import csv
 rows=list(csv.DictReader((ROOT.parent/"data/train_index.csv").open()))
 gt=[np.load(ROOT.parent/f"data/train_pl{x}.npy",mmap_mode="r") for x in ("415","58")]
 masks=[np.load(ROOT.parent/f"data/train_mask{x}.npy",mmap_mode="r") for x in ("415","58")]
 pred=np.load(ROOT/"runs/c155_unet/selected_monitor_prediction.npy",mmap_mode="r")
 c82=np.load(ROOT/"audit/c88_mesh/base.npy",mmap_mode="r");cidx={v:i for i,v in enumerate(read(ROOT/"audit/c88_mesh/base_index.json")["ids"])}
 for c in d["top3_regions"]:
  a=np.load(ROOT/c["data"]);idx=c["map_id"];j=p["ordered_ids"].index(idx)
  np.testing.assert_array_equal(a["GT"],np.stack([x[idx] for x in gt]).astype(float))
  np.testing.assert_array_equal(a["valid"],np.stack([np.unpackbits(x[idx],bitorder="big").reshape(256,256) for x in masks]).astype(bool))
  np.testing.assert_array_equal(a["C155"],pred[j]);np.testing.assert_array_equal(a["C82"],c82[cidx[idx]])
  err82=a["C82"].astype(float)-a["GT"];err155=a["C155"].astype(float)-a["GT"]
  expected=a["mask"][None]&a["valid"]&(np.abs(err82)>=5)&(np.abs(err155)>=5)
  np.testing.assert_array_equal(expected,a["common_tail"])
  close(np.where(expected,.5*(err82**2+err155**2),0).sum(),c["joint_SSE_dB2"],atol=1e-6)
  begin,end=map(int,z["offsets"][j:j+2]);pixels=z["pixels"][begin:end];values=z["counts"][begin:end]
  for b,stats in enumerate(c["per_band"]):
   mm=expected[b];assert int(mm.sum())==stats["count"]
   for model,key in [("GT","GT"),("C82","C82"),("C155","C155"),("FSPL_Deygout","Deygout"),("FSPL","FSPL")]:
    v=a[key][b].astype(float);e=v[mm]-a["GT"][b][mm]
    close(v[mm].mean(),stats["models"][model]["mean_dB"],atol=2e-5)
    close(np.median(e),stats["models"][model]["median_difference_vs_GT_dB"],atol=2e-5)
    close(np.sqrt(np.mean(e*e)),stats["models"][model]["RMSE_vs_GT_dB"],atol=1e-6)
   pix=np.flatnonzero(mm);positions=np.searchsorted(pixels,pix)
   np.testing.assert_array_equal(pixels[positions],pix)
   close(values[positions,0].mean(),stats["mean_n"]);close(values[positions,1].mean(),stats["mean_blocked_segments"])
  h=a["profile_height_m"];ray=a["profile_direct_ray_m"];st=h-ray>1e-10
  n=int(np.count_nonzero(st[1:]!=st[:-1]));blocks=int(st[0])+int(np.count_nonzero(st[1:]&~st[:-1]))
  assert n==c["profile_crossings"] and blocks==c["profile_blocked_segments"]
  cases_verified.append({"rank":c["rank"],"sample_id":c["sample_id"],"counts":[s["count"] for s in c["per_band"]],"independent_direct_arrays":True,"geometry_profile_n":n,"geometry_profile_blocked_segments":blocks,"case_sha256":sha(ROOT/c["data"])})
 t=read(OUT/"tracking.json");folder=Path(t["dir"])
 db=sqlite3.connect("file:"+str(ROOT/"tracking/swanlab/logs/runs.swanlab")+"?mode=ro",uri=True);db.row_factory=sqlite3.Row
 experiment=dict(db.execute("select * from experiment where run_id=?",(folder.name,)).fetchone())
 project=dict(db.execute("select * from project where id=?",(experiment["project_id"],)).fetchone())
 tags=[dict(x) for x in db.execute("select * from tag where experiment_id=?",(experiment["id"],))]
 expected_metrics={}
 for scope in ("all","C82_good"):
  for stage in d["scores"][scope]:
   for b,band in enumerate(("415MHz","5p775GHz")):
    expected_metrics[f"{scope}/{stage}/RMSE_{band}_dB"]=d["scores"][scope][stage]["RMSE_dB"][b]
 for b,band in enumerate(("415MHz","5p775GHz")):
  expected_metrics[f"common_tail/{band}/count"]=common["count"][b]
  expected_metrics[f"common_tail/{band}/mean_n"]=common["crossing_count_n_mean"][b]
 logged={}
 for tag in tags:
  name=tag["name"]
  if name not in expected_metrics:continue
  logs=folder/"logs"/tag["folder"]
  assert logs.exists(),(tag,logs)
  points=[json.loads(line) for file in logs.glob("*.log") for line in file.read_text().splitlines() if line.strip()]
  point=next(x for x in points if x["index"]==629)
  assert float(point["data"])==float(expected_metrics[name]),(name,point,expected_metrics[name])
  logged[name]=point["data"]
 assert len(logged)==len(expected_metrics)==36,(len(logged),len(expected_metrics),[(x["sort"],x["folder"],x["name"]) for x in tags])
 assert project["name"]=="Lunar_CC"
 config=(folder/"files/config.yaml").read_text();assert "mode: local" in config
 frozen_before={x:sha(OUT/x) for x in ("protocol.json","result.json","crossing_counts_sparse.npz","case_1.npz","case_2.npz","case_3.npz")}
 figfile=ROOT/"src/c156_figures.py";old=figfile.read_text()
 if 'ax.set_title("同一行的额外衰减")' in old:
  changed=old.replace('ax.set_title("同一行的额外衰减")','ax.set_title(("415 MHz" if b==0 else "5.775 GHz")+"：同一行的额外衰减")')
  revisions=[{"file":"src/c156_figures.py","old_sha256":sha(figfile),"reason":"Add explicit frequency to representative line profiles; display only; no data or statistics changed"}]
  figfile.write_text(changed);revisions[0]["new_sha256"]=sha(figfile)
 else:revisions=existing["presentation_revisions"]
 from src.c156_report import run as make_report
 make_report()
 text=(OUT/"report.md").read_text();start=text.index("## C155到C82的平方误差收益分摊");end=text.index("## 空间偏移、尺度与平滑诊断",start)
 segment=["## C82分量的边际改善（固定输出，不是训练消融）","",
  "|变换|全图MSE降低% 415/5.775|good区MSE降低% 415/5.775|","|---|---|---|"]
 for key,label in [("base -> local","base → base+local"),("base -> graph","base → graph（并行）"),("graph -> blend","graph → 50/50融合"),("blend -> calibration_replay","融合 → 多尺度校准")]:
  a=stages["all"][key]["MSE_reduction_percent"];b=stages["C82_good"][key]["MSE_reduction_percent"]
  segment.append(f"|{label}|{a[0]:.2f} / {a[1]:.2f}|{b[0]:.2f} / {b[1]:.2f}|")
 segment.extend(["","局部与图网络是两条并行分支；两条base对照不能相加。高精度基本盘主要来自两条学习修正及互补融合，多尺度校准是最后的小幅改善。",
  "result.json保留了以C155为起点的二次交互等分分摊。该分摊依赖参照与分解方式，可能出现>100%和负份额；不能据此声称local/graph有害。这里用固定实际阶段的边际收益解释。",
  "",""])
 text=text[:start]+"\n".join(segment)+text[end:]
 text += ("\n## 核验后的解释\n\n"+
  "C155好区退化主要表现为随位置变化的局部均值误差。±2像素统一整数平移最优为原位；全局仿射最多消除约4%误差能量；简单高通增益最多约1.4%，不支持仅靠整体对齐、缩放或锐化解决。该诊断不排除更小尺度或空间变化的位移。\n"+
  "共同大错并非统一的多重深绕射：约43%–46%没有直达线地形遮挡；多于一个遮挡段只占18.77%/6.70%。06501_05高频和05608_00低频主要衰减过强，01128_45高频则是单段深阴影衰减不足。图像外圈略富集，TX近距未富集；全局坡度与局部地形高通也没有强升高，不支持把所有错误归为地形突变或边缘。\n"+
  "Top3按当前输出联合SSE排名抽样；01128_45低频仅1个共同大错像素，该行不能推广到整个低频区域。C82_good由C82选定；monitor曾参与C82校准，属于开发诊断，不是盲测或因果机制证明。\n")
 (OUT/"report.md").write_text(text)
 from src.c156_figures import render
 render()
 for f,h in frozen_before.items():assert sha(OUT/f)==h,(f,"changed")
 verification={"status":"numerical_checks_passed_awaiting_revised_figure_inspection","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "source_sha256":sha(Path(__file__)),"frozen_core_artifacts_sha256":frozen_before,"input_identities_unchanged":True,
  "protected_parent_identities_unchanged":True,"C82_prediction_full_SHA_unchanged":True,"C155_selected_weight_and_monitor_prediction_full_SHA_unchanged":True,
  "parent_metrics_match":True,"score_algebra_class_support_and_shift_checks":True,"stage_marginal_gains":stages,"case_checks":cases_verified,
  "presentation_revisions":revisions,"SwanLab":{"mode":"local","run_id":t["run_id"],"native_experiment_id":experiment["id"],"project":project["name"],"final_metrics_verified":len(logged),"step":629,"raw_values_exact":True},
  "training":False,"neural_inference":False,"GPU_used":False}
 write(OUT/"verification.json",verification)
 print(json.dumps({"status":verification["status"],"SwanLab":verification["SwanLab"],"stage_gains_good":stages["C82_good"],"cases":cases_verified},ensure_ascii=False))
if __name__=="__main__":run()
