"""Consolidate actual C47-C65 results, with evaluation scopes kept explicit."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((R/p).read_text())
proof=read('audit/c65_final_delivery.json');new=proof['official'];old=proof['C38_official']
names=['c47_flat_full','c47_conv_full','c47_attention_full','c48_patch_full','c48_both_full','c49_unweighted_full','c49_transport_full','c50_cooperative_full','c52_adapter_full','c52_transport_full','c54_low_attention','c54_high_transport','c56_multibounce_full','c59_narrow1024','c59_wide2048','c61_local_full','c61_path_full','c63_fixed_full','c63_dynamic_full']
runs={n:read('runs/'+n+'/result.json') for n in names if (R/'runs'/n/'result.json').exists()}
c54=read('audit/c54_band_experts_20261102/c54_combo_dev_compiled_all_score.json')
coverage=read('audit/c59_capacity_coverage_20261102/coverage.json')
tiny=read('audit/c61_global_terrain_20261102/tiny_unseen_pixels.json')
fixed=read('runs/c62_fixed_4maps_continue3000/result.json');dense=read('runs/c62_dense_4maps_continue3000/result.json')
comparison=read('audit/c63_dynamic_full_20261102/comparison.json');phase=read('audit/c64_height_phase_20261102/result.json')
runtime={h:read('audit/c65_full_refit_20261102/interface_'+h+'.json') for h in ['H100','A800']}
summary={'date':'2026-11-02','goal':'Both official fixedVAL band maskedRMSE<=1dB','goal_reached':proof['goal_reached'],'official_C38':old,'official_C65':new,'C65_delivery':proof,'complete_internalDEV612_C54':c54,'sampled_DEV612_runs':runs,'C38_pixel_coverage':{k:v for k,v in coverage.items() if k!='permap'},'C61_tiny_pixel_generalization':{k:v for k,v in tiny.items() if k!='permap'},'C62_fixed':fixed,'C62_dense':dense,'C63_comparison':comparison,'C64_height_phase':phase,'runtime':runtime,'source_root':str(R),'official_VAL_training':False,'website_submission':False}
(R/'results/C47_C65_results_20261102.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
fmt=lambda a:' / '.join(f'{v:.6f}' for v in a)
rows=[]
for n,r in runs.items():
 if 'DEV' in r and 'RMSE' in r['DEV']:rows.append(f"| {n} | {fmt(r['DEV']['RMSE'])} | {r.get('best_step','—')} / {r.get('last_step','—')} |")
common=next(r for r in comparison['shared_steps'] if r['step']==5000);g=proof['gain_vs_C38_db']
csvpath=str(R/'submissions/C65_Kaggle_20261102/submission_C65.csv') if proof.get('CSV') else '未生成推荐上传CSV'
report=f"""# C47–C65试验与正式验证

本轮未做出用户要求的大幅提分版本。双频均≤1dB目标判定：{proof['goal_reached']}。
已完成架构、功率传播、采样密度及候选相位对照；冻结C65后完成全TRAIN再拟合和完整官方固定VAL计分。
所有实现、训练、评估均在{R}独立完成。只用实体1/3/4完整UUID。没有网站提交。

## 正式结果

| 模型 | 415MHz RMSE | 5.775GHz RMSE | 双频平均 |
| --- | ---: | ---: | ---: |
| C38 | {old['RMSE'][0]:.9f} | {old['RMSE'][1]:.9f} | {old['combined_rmse_db_masked']:.9f} |
| C65 | {new['RMSE'][0]:.9f} | {new['RMSE'][1]:.9f} | {new['combined_rmse_db_masked']:.9f} |
| 改善dB | {g[0]:.9f} | {g[1]:.9f} | {old['combined_rmse_db_masked']-new['combined_rmse_db_masked']:.9f} |

C65是否共同优于C38：{proof['jointly_better_than_C38']}。两个频段要分别≤1dB，不能用平均分替代目标。
范围：680块地形、每频段2330图，有效像素151537682 / 135926253。
独立CPU float64计分与官方原版score_submission.py、metrics.py相符，差小于1e-5dB。
复用此前已核验的原版官方代码，固定版本fa3ef9b89f964556753a18be102a5aadeb723377。
综合值为分频RMSE的算术平均；不是跨队归一化排名分，也不是新的网站分数或隐藏测试结果。
证据：{R}/audit/c65_combo_full_official/verified_summary.json。

## 最终方法与数据隔离

C65采用全TRAIN C38周期数值嵌入/8成员BatchEnsemble式MLP作为FSPL残差基底。
低频叠加128点有序地形剖面的注意力残差；高频叠加16方向×32半径地表节点的功率传输注意力残差。
高频注意力使用可见性、入射/出射角和几何功率先验。它不是恢复出的真实路径或PDP/PAS。
按频段固定组合，所有像素都使用相同规则。未采用本轮验证失败的C61/C63整图CNN。

内部开发只用5508块FIT地形、23158图训练，612块DEV地形、2562图选模型。
冻结C54固定组合在完整内部DEV612上为{fmt(c54['C54_RMSE'])}dB；
C38同面板{fmt(c54['C38_RMSE'])}dB，改善{fmt(c54['gain_db'])}dB。
两频段地形配对bootstrap均支持正收益，但不是大幅提分。
较小DEV64高频只改善0.083449dB，未过0.10dB门槛；只补做完整DEV612推理后，双频都超过原0.10dB门槛，未降低目标。
内部DEV已反复用于候选选择，不能称作全新独立测试。

C65使用全部官方TRAIN：6120块地形、25720图，每图1024点，包含原内部DEV。
它从冻结的C54专家温启动，以全TRAIN C38为共同基底；不是从随机初始权重完整重跑。
低频batch4096、高频batch2048，学习率0.0003、余弦日程、weight decay1e-4。
两个残差头各固定17770更新，由16000×25720/23158预先换算。最后一步取权重，不用官方VAL早停。
官方VAL未用于梯度、选步数或路由。历史C38工作曾用官方VAL作事后评分与诊断，不能说从未看过VAL。
本轮权重、路由冻结后，H/TX输入推理不读取标签或掩码；完成预测后才统一计分。
官方VAL结果已用于C65与C38的最终对比及交付判断，不宣称它是从未用于版本判断的独立隐藏测试。
掩码只用于训练损失和评分，不作为输入。

## 全部主要内部候选

以下均为相同内部DEV612上的655872个采样位置，频段有效数650982 / 596173。
C38为2.644635 / 2.998740dB。各行预算不同，不将整个表当成严格单变量消融。

| 运行 | 415MHz / 5.775GHz dB | 选中步 / 末步 |
| --- | --- | --- |
{chr(10).join(rows)}

C54固定组合选c54_low_attention第0头与c54_high_transport第1头；
该整体模型同采样为2.526593 / 2.884798dB，不是逐像素按标签挑结果。
C50未改善；C52适配器与C56粗多跳算子未产生足以改变总体结论的收益。
C57/C58受限传输近似的低频确认收益约0.0042/0.0044dB，没有据此追加训练。
不能用这些近似的失败证明物理表征无用或给出不可约误差下限。

## 小集拟合与采样密度的证据

冻结C38在64块FIT地形全部264图中，已见1024点{fmt(coverage['groups']['seen1024']['RMSE'])}dB，
同图其余像素{fmt(coverage['groups']['unseen64512']['RMSE'])}dB。C38本身的这项差距相对有限。

C61路径版在四块地形4096已知点达到{fmt(tiny['groups']['seen1024']['C61']['RMSE'])}dB，
但同地形未见像素为{fmt(tiny['groups']['unseen64512']['C61']['RMSE'])}dB，
差于C38同像素{fmt(tiny['groups']['unseen64512']['C38']['RMSE'])}dB。
小集训练点拟合不等于学会细尺度传播结构。

C62保持网络、初始参数、地图顺序、预算相同。每图4096预留像素，两组均不训练这些目标。
固定组只在1024点采样；密集组在其余61440点采样。总4000更新后：

| 同四块已知地形的预留像素 | 415MHz / 5.775GHz dB |
| --- | --- |
| C38 | {fmt(dense['holdout4096']['C38_RMSE'])} |
| 固定1024点 | {fmt(fixed['holdout4096']['RMSE'])} |
| 密集61440点 | {fmt(dense['holdout4096']['RMSE'])} |

这支持修复固定点记忆，但尚不支持跨地形大幅提分。
因此进行C63全FIT固定/动态采样对照，初始权重哈希相同，特征重建最大差4.58e-6。
重建特征对C38预测RMS影响0.000238/0.000345dB。
相同5000步：固定{fmt(common['fixed_DEV'])}，动态{fmt(common['dynamic_DEV'])}dB。
动态最终6000步为{fmt(runs['c63_dynamic_full']['DEV']['RMSE'])}dB，未过双频各0.15dB扩展门槛。
固定组按共同耐心规则在5000步结束，最佳3000步；不能把不等末步差全部归因于采样方式。
动态每步从65536像素空间重采样；6000步约2458万次目标采样，不代表遍历全部像素。
不继续延长该分支。

## 仍未闭合的问题

9组相同H/TX的标签差0.020/0.067dB只说明该重复子集一致，不是独立种子噪声估计。
C64受限高度相位反事实在FIT选择零偏移；跨四个多TX地形的留出TX未改善。
不支持继续把一个地形统一相位偏移作为主要修复方向；该诊断不是实际TX或原网格恢复。
原始传播网格、实际连续TX、插值前单元功率和逐种子结果仍未恢复。
不能把剩余误差全部归因于标签错配、输入不足、随机噪声或单纯网络太小。

## 推理与交付

H100两张FIT检查图预热双频耗时：{fmt([r['warm_seconds_both_bands'] for r in runtime['H100']['records']])}秒。
A800对应{fmt([r['warm_seconds_both_bands'] for r in runtime['A800']['records']])}秒，未通过双频总计500ms的保守检查。
首次编译另计，官方硬件未测量。评分改变量小于0.001dB的精度检查只覆盖两张FIT图；
正式结果来自实际保存预测的完整计分。

服务器复现包：{proof.get('package') or '未打包为共同更优版本'}。
CSV：{csvpath}。
CSV模板对应100个公开VAL子集样本，不使用真实标签或掩码填值，80万行与原始推理分片float32逐值核对。
CSV不是隐藏测试最终平台模型包。未进行网站提交，旧C38上传文件保留。
完整指标、配置、来源和正负结果见同目录C47_C65_results_20261102.json。
交付校验见{R}/audit/c65_final_delivery.json。

C61最终指标保存曾遇到numpy布尔序列化问题；训练、权重、历史和预测完整，
已从保存预测独立CPU float64复算恢复，没有重训或补造数据。全部失败日志保留在CC。
"""
(R/'results/C47_C65实验与正式验证_20261102.md').write_text(report)
print(json.dumps({'report':str(R/'results/C47_C65实验与正式验证_20261102.md'),'RMSE':new['RMSE'],'goal_reached':proof['goal_reached'],'CSV':bool(proof.get('CSV'))},ensure_ascii=False),flush=True)
