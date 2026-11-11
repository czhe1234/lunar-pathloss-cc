"""Verify and record the exact public scorer result on all fixed VAL maps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];O=R/'audit/c10_official_scorer_20261101'
read=lambda p:json.loads(p.read_text())
official=read(O/'official_report.json');provenance=read(O/'provenance.json');ref=provenance['expected_reference']
assert len(official['per_sample'])==4660
checks={};band_results={}
for b in ['415','58']:
 rows=[r for r in official['per_sample'] if r['band']==b]
 assert len(rows)==2330 and len({r['sample_id'] for r in rows})==2330
 n=sum(int(round(r['valid_frac']*65536)) for r in rows)
 assert n==ref['bands'][b]['valid_pixels']==provenance['valid_pixels'][b]
 value=official['per_band'][b]['rmse_db_masked'];difference=abs(value-ref['bands'][b]['rmse_db'])
 assert difference<1e-5
 band_results[b]={'official_script_masked_RMSE_db':value,'original_FP64_masked_RMSE_db':ref['bands'][b]['rmse_db'],'absolute_difference_db':difference,'maps':len(rows),'valid_pixels':n}
 checks[b+'_complete_and_matches_independent_score']=True
for name,s in provenance['sources'].items():
 body=(O/'official'/name).read_bytes()
 assert hashlib.sha256(body).hexdigest()==s['sha256'] and hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()==s['git_blob']
checks['official_sources_unmodified_and_hash_verified']=True
combined=official['summary']['combined_rmse_db_masked']
assert combined==sum(r['official_script_masked_RMSE_db'] for r in band_results.values())/2
result={'model':'C10','scope':'complete official fixed validation set, 2330 maps per band; not withheld test','official_script':'score_submission.py','official_repository_revision':provenance['official_repository_revision'],'metric_key':'combined_rmse_db_masked','combined_RMSE_db':combined,'per_band':band_results,'checks':checks,'GPU_used':False,'training_performed':False,'website_submission':False,'final_cross_team_normalized_rank_score':None,'rank_note':official['summary']['ranking_note'],'official_report':str(O/'official_report.json'),'provenance':str(O/'provenance.json')}
(O/'verified_summary.json').write_text(json.dumps(result,indent=2)+'\n')
report=f"""# C10 官方原版评分脚本复算

综合指标 combined_rmse_db_masked：**{combined:.10f} dB**，越低越好。

| 项目 | 结果 |
|---|---:|
| 415 MHz masked RMSE | {band_results['415']['official_script_masked_RMSE_db']:.10f} dB |
| 5.8 GHz masked RMSE | {band_results['58']['official_script_masked_RMSE_db']:.10f} dB |
| 双频50/50综合RMSE | {combined:.10f} dB |

直接执行官方公开仓库的原版 score_submission.py 和 metrics.py。
固定版本：{provenance['official_repository_revision']}。
源码Git blob和SHA256均核验，评分代码未修改。

评价范围是完整官方固定验证集：每频段2,330图；有效像素151,537,682 / 135,926,253。
已保存的C10预测、原始验证标签和有效掩码转换为官方脚本要求的逐图文件布局。
脚本要求目录名test；该目录实际装的是验证集，未改称隐藏测试集。
没有训练、模型推理、GPU运算或网站提交。

官方脚本按各频段全部有效像素累计平方误差，算出各自RMSE，然后做50/50平均。
官方原版分项结果与此前独立FP64计算的差异为
{band_results['415']['absolute_difference_db']:.12g} / {band_results['58']['absolute_difference_db']:.12g} dB。
这个数值差异处于浮点计算精度范围内。

官方脚本同时明确：最终排名还要对全部参赛队伍的分频RMSE归一化后合并。
单个参赛结果无法计算这个排名分。因此本次得到官方算法的验证集综合RMSE，尚无跨队归一化排名分。

- 官方脚本：https://github.com/anderspearson206/lunar-radiomap-challenge/blob/{provenance['official_repository_revision']}/score_submission.py#L223
- 原始评分输出：{O}/official_report.json
- 核验汇总：{O}/verified_summary.json
- 来源和转换记录：{O}/provenance.json
- 运行日志：{R}/logs/c10_official_scorer.log
"""
rp=R/'results/C10官方评分脚本复算_20261101.md';rp.write_text(report)
p=R/'audit/c10_final_delivery.json';d=read(p);d['official_script_replay']={'combined_RMSE_db':combined,'summary':str(O/'verified_summary.json'),'report':str(rp)};p.write_text(json.dumps(d,indent=2)+'\n')
p=R/'STATUS.md';s=p.read_text();marker='## 官方评分脚本复算'
assert marker not in s
pos=s.index('\n\n下文为已完成 C5')
note=f'\n\n{marker}\n\n官方原版score_submission.py复算全部固定验证图，combined_rmse_db_masked={combined:.10f} dB。\n源码和有效像素数核验通过，分项与独立FP64计算差异均小于1e-5 dB。\n这是验证集综合RMSE；跨队归一化排名需要其他参赛队伍成绩。报告：results/C10官方评分脚本复算_20261101.md。'
p.write_text(s[:pos]+note+s[pos:])
p=R/'results/C10新版本与官方验证_20261101.md';s=p.read_text();s+=f'\n## 官方原版评分脚本复算\n\n直接运行官方score_submission.py所得综合RMSE为 **{combined:.10f} dB**。\n完整核验与范围说明见{rp}。\n';p.write_text(s)
print(json.dumps(result),flush=True)
