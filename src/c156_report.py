"""C156 tables and local SwanLab recording from completed CPU statistics."""
import json,os
import numpy as np
from src.c156_data import ROOT,OUT,STAGES,BANDS,load,sha,atomic
def f(v,n=4):return "—" if v is None else f"{v:.{n}f}"
def pair(v,n=4):return " / ".join(f(x,n) for x in v)
def run():
 d=load(OUT/"result.json");p=load(OUT/"protocol.json")
 assert d["status"]=="complete" and d["source_files_and_identities_unchanged"] and not d["CUDA_initialized"]
 names={"base":"base","local":"base + local correction","graph":"graph（独立并行）","blend":"50/50 local + graph",
  "calibration_replay":"blend + 多尺度校准（CPU重组）","C82":"最终冻结 C82","C155":"C155 selected52","FSPL_Deygout":"FSPL + L_Deygout"}
 lines=["# C156：纯 CPU 分量归因与共同大错区核查","",
  "628图/128地形；所有分量均有现成缓存。没有训练、神经网络推理、GPU计算或传播功率重算。",
  "C82_good=valid且|C82−GT|≤1；共同大错=valid且|C82−GT|≥5且|C155−GT|≥5，逐频定义。","",
  "## 分量输出：固定同一像素集合","",
  "|输出|全图415|全图5.775GHz|C82_good415|C82_good5.775GHz|","|---|---:|---:|---:|---:|"]
 for s in STAGES:
  a=d["scores"]["all"][s]["RMSE_dB"];g=d["scores"]["C82_good"][s]["RMSE_dB"]
  lines.append(f"|{names[s]}|{f(a[0])}|{f(a[1])}|{f(g[0])}|{f(g[1])}|")
 lines+=["","全部值为masked RMSE/dB。graph为并行专家，不是local之后的串行阶段。C82_good由最终C82选定，存在条件化选择效应；必须结合全图列。","",
  "CPU校准重组与冻结C82的差异：RMSE "+pair(d["calibration_replay_residual"]["RMSE_dB"],8)+" dB；maxAbs "+pair(d["calibration_replay_residual"]["max_abs_dB"],8)+" dB。",
  "差异被单独列入cache/replay残差，没有强行计入多尺度校准收益。","",
  "## C155到C82的平方误差收益分摊","",
  "固定分量的二次交互项等分，逐项和严格等于实际SSE差；这是输出归因，不是重新训练的因果消融。",
  "|项|全图 MSE收益415/5.775|good区MSE收益415/5.775|good区收益占比% 415/5.775|","|---|---|---|---|"]
 a=d["C155_excess_MSE_attribution"]
 for j,name in enumerate(a["all"]["terms"]):
  lines.append(f"|{name}|{pair(a['all']['MSE_gain_dB2'][j],6)}|{pair(a['C82_good']['MSE_gain_dB2'][j],6)}|{pair(a['C82_good']['percent_of_C155_to_C82_SSE_gain'][j],2)}|")
 diag=d["C82_good_diagnostics"];sh=diag["integer_shift"]
 lines+=["","## 空间偏移、尺度与平滑诊断","",
  "以下oracle仅为GT辅助的数学诊断，未修改预测；收益不是新验证成绩。整数平移比较固定2像素内圈good支持域，所有平移完全同一批GT像素。","",
  "|频段|原位内圈 RMSE|最佳统一(dy,dx)|统一平移后 RMSE|MSE下降%|逐图平移oracle RMSE|","|---|---:|---|---:|---:|---:|"]
 for b,band in enumerate(BANDS):
  v=sh["best_global"][b]
  lines.append(f"|{band}|{f(sh['identity_RMSE_dB'][b])}|({v['dy']},{v['dx']})|{f(v['RMSE_dB'])}|{f(v['MSE_reduction_percent'],2)}|{f(sh['per_image_oracle_RMSE_dB'][b])}|")
 lines+=["","|全局两标量oracle|a 415/5.775|b 415/5.775|RMSE 415/5.775|MSE下降% 415/5.775|","|---|---|---|---|---|"]
 for k,v in diag["affine_oracles"].items():
  lines.append(f"|{k}|{pair(v['a'])}|{pair(v['b'])}|{pair(v['oracle_RMSE_dB'])}|{pair(v['MSE_reduction_percent'],2)}|")
 lines+=["","|支持域/模型|GT高通RMS415/5.775|Pred高通RMS415/5.775|高通误差RMS415/5.775|局部均值误差RMS415/5.775|","|---|---|---|---|---|"]
 for support,models in diag["roughness"].items():
  for model,v in models.items():
   lines.append(f"|{support}/{model}|{pair(v['GT_minus_FSPL_highpass_RMS_dB'])}|{pair(v['Pred_minus_FSPL_highpass_RMS_dB'])}|{pair(v['highpass_error_RMS_dB'])}|{pair(v['localmean_error_RMS_dB'])}|")
 lines+=["","|高通增益oracle支持域|alpha 415/5.775|最优RMSE 415/5.775|MSE下降% 415/5.775|固定去高通(alpha=-1)RMSE|","|---|---|---|---|---|"]
 for k,v in diag["highpass_oracle"].items():
  lines.append(f"|{k}|{pair(v['alpha_optimal'])}|{pair(v['oracle_RMSE_dB'])}|{pair(v['MSE_reduction_percent'],2)}|{pair(v['fixed_smoothing_alpha_minus1_RMSE_dB'])}|")
 lines+=["","HP/LP不是正交分解，报告保留cross term，不能直接把两项MSE相加当作总MSE。","",
  "## 共同大错区","",
  "|频段|C82原大错占全部像素%|两者共同大错占全部%|共同区C82 RMSE|共同区C155 RMSE|共同区占C155总SSE%|","|---|---:|---:|---:|---:|---:|"]
 alln=np.asarray(d["scores"]["all"]["C155"]["count"]);alls=np.asarray(d["scores"]["all"]["C155"]["SSE_dB2"])
 for b,band in enumerate(BANDS):
  tail=d["scores"]["C82_tail"]["C155"];co=d["scores"]["common_tail"]
  lines.append(f"|{band}|{100*tail['count'][b]/alln[b]:.3f}|{100*co['C155']['count'][b]/alln[b]:.3f}|{f(co['C82']['RMSE_dB'][b])}|{f(co['C155']['RMSE_dB'][b])}|{100*co['C155']['SSE_dB2'][b]/alls[b]:.2f}|")
 lines+=["","|共同大错中的类别|像素占比%415/5.775|平均n 415/5.775|平均遮挡段415/5.775|平均有限ν415/5.775|","|---|---|---|---|---|"]
 for g in ["common_tail","common_Clear","common_Grazing","common_NLoS"]:
  v=d["physical"][g];den=np.asarray(d["physical"]["common_tail"]["count"])
  lines.append(f"|{g}|{pair((100*np.asarray(v['count'])/den).tolist(),2)}|{pair(v['crossing_count_n_mean'])}|{pair(v['blocked_segments_mean'])}|{pair(v['mean']['nu'])}|")
 lines+=["","n是当前注册几何下直达线穿越地形的次数，不是反射次数，也不等于山脊数量；遮挡段数另列。原c73平均log次数没有被错误地反解成平均n。","",
  "|条件|全部有效像素占比%415/5.775|共同大错占比%415/5.775|富集倍数415/5.775|","|---|---|---|---|"]
 common=d["physical"]["common_tail"];allphys=d["physical"]["all"]
 for key,label in [("total_floor_percent","c73总功率floor编码"),("near_TX_lt10m_percent","TX距离<10m"),("far_ge150m_percent","TX距离≥150m"),("outer4pixel_percent","图像外圈4像素")]:
  base=np.asarray(allphys[key]);v=np.asarray(common[key]);enrich=np.divide(v,base,out=np.full(2,np.nan),where=base>0)
  lines.append(f"|{label}|{pair(base.tolist(),3)}|{pair(v.tolist(),3)}|{pair([float(x) if np.isfinite(x) else None for x in enrich],2)}|")
 lines+=["","floor是Q=P·d_face²的压缩值−3，非绝对功率P触底。近/远仅为距离分类，不能据此称电磁近场/远场。完整距离分桶、n/遮挡段直方图、ν分位数见result.json。","",
  "## Top-3区域：固定联合SSE排名","",
  "|区域/频段|共同像素|GT均值|C82均值|C155均值|FSPL+Ld均值|C82−GT中位数|C155−GT中位数|FSPL+Ld−GT中位数|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for case in d["top3_regions"]:
  for v in case["per_band"]:
   z=v["models"]
   lines.append(f"|{case['rank']} {case['sample_id']} {v['band']}|{v['count']}|{f(z['GT']['mean_dB'])}|{f(z['C82']['mean_dB'])}|{f(z['C155']['mean_dB'])}|{f(z['FSPL_Deygout']['mean_dB'])}|{f(z['C82']['median_difference_vs_GT_dB'])}|{f(z['C155']['median_difference_vs_GT_dB'])}|{f(z['FSPL_Deygout']['median_difference_vs_GT_dB'])}|")
 lines+=["","|区域/频段|Clear/Grazing/NLoS %|平均n|平均遮挡段|平均ν|floor %|典型剖面n/段|","|---|---|---:|---:|---:|---:|---|"]
 for case in d["top3_regions"]:
  for v in case["per_band"]:
   lines.append(f"|{case['rank']} {case['sample_id']} {v['band']}|{pair(v['class_percent'],2)}|{f(v['mean_n'])}|{f(v['mean_blocked_segments'])}|{f(v['mean_nu'])}|{f(v['floor_percent'],2)}|{case['profile_crossings']}/{case['profile_blocked_segments']}|")
 for case in d["top3_regions"]:
  lines+=["",f"区域{case['rank']}：bbox={case['bbox_rows_cols']}，联合SSE={case['joint_SSE_dB2']:.3f}；near<10m%={case['near_TX_lt10m_percent']}；far≥150m%={case['far_ge150m_percent']}；外圈%={case['outer4pixel_percent']}。",
   f"![区域{case['rank']}](region_{case['rank']}.png)"]
 lines+=["","多重山脊、局部突变和发布边缘只能按当前几何/画像证据区分支持度；此统计不强行把未知机制归为唯一原因。","",*d["limits"]]
 (OUT/"report.md").write_text("\n".join(lines)+"\n")
 return d
if __name__=="__main__":run()
