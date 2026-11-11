from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c125_tail';B=R/'audit/c126_statefield'
diag=json.load(open(A/'diagnosis.json'));capacity=json.load(open(A/'capacity_verified.json'));view=json.load(open(A/'neural_comparison.json'));gate=json.load(open(A/'frozen_gate_probe.json'));score=json.load(open(B/'score.json'))
tree={m:[json.load(open(R/f'runs/c125_tree_{m}/result{b}.json')) for b in range(2)] for m in ['old','native']}
state=json.load(open(R/'runs/c126_state/result.json'));risk=json.load(open(R/'runs/c126_risk/result.json'))
d=diag['bands'];v=view['bands'];g=gate['bands'];s=score['models']
lines=['# C125–C126：大误差点的方向、幅度与状态识别（2026-11-08）','',
'结论：本轮没有形成大幅提分版本。55% 不是现有输入的信息上限；它是全点回归器在真实大误差子集上的方向表现。专门训练可以改善子集方向，但仍未学会在新地形上准确判断大误差的位置、符号及幅度。训练方向可被记忆，不能转化成跨地形收益。现有证据不能判定原始地形/TX 必然缺少关键物理信息。','',
f"针对性修改后的完整地形网络，在同一 TRAIN 留出集的完整有效像素上为 **{s['state']['RMSE'][0]:.6f} / {s['state']['RMSE'][1]:.6f} dB**，相对 C82 仅改善 **{s['state']['gain_vs_C82'][0]:.6f} / {s['state']['gain_vs_C82'][1]:.6f} dB**。不进入确认集，不扩训，不替换正式提交。官方固定 VAL 的 C82 仍为 2.246377 / 2.579150 dB，双频 1 dB 目标未完成。",'',
'所有源文件、计算、模型和缓存位于 transformer:/home/chenzhe/work/lunar-pathloss-rm-challenge/CC。本地只提供核验后的交付副本。仅使用当时空闲的授权 1、3、4 号卡。','',
'## 1. 55% 究竟代表什么','',
'固定 C101 monitor 的相同大误差点：有效且 |Y−C82|≥5 dB。低频 47,962 点，高频 53,681 点，占各频段有效采样点 3.76% / 4.54%。Y 是路径损耗；正残差表示需要增加预测路径损耗。','',
'| 方向预测方式 | 415 MHz | 5.775 GHz |','|---|---:|---:|']
for label,vals in [('C124 全点回归器，在大误差子集计分',[z['conditional']['sign_accuracy'] for z in d]),('总是预测子集多数方向',[z['majority_accuracy'] for z in d]),('C120 旧169表征，专门方向分类',[z['C120_old']['conditional_sign_accuracy'] for z in d]),('C120 加入原生地形潜在特征，专门方向分类',[z['C120_native']['conditional_sign_accuracy'] for z in d])]:
 lines.append(f'| {label} | {vals[0]*100:.2f}% | {vals[1]*100:.2f}% |')
lines+=['','专门分类器的输入仍只有地形/TX 推导的特征。但它只在真实大误差点上训练和报告方向结果；它没有解决全图上哪些点适合应用修正，也没有给出足够准确的幅度。方向准确率不能直接替代 RMSE。','',
'全点 MSE 回归的目标是给定输入的平均残差。不同误差状态尚未分清时，正负修正会相互抵消。专门训练的提升说明训练目标有影响；不能把 55% 解读成“输入只能包含55%的信息”。','',
'## 2. 方向与幅度必须一起解决','',
'| C124 在真实大误差点上的中位数 | 415 MHz | 5.775 GHz |','|---|---:|---:|']
lines.append(f"| 需要修正的绝对幅度 | {d[0]['conditional']['target_abs_q50_q90_q99'][0]:.4f} dB | {d[1]['conditional']['target_abs_q50_q90_q99'][0]:.4f} dB |")
lines.append(f"| 模型实际修正的绝对幅度 | {d[0]['conditional']['predicted_abs_q50_q90_q99'][0]:.4f} dB | {d[1]['conditional']['predicted_abs_q50_q90_q99'][0]:.4f} dB |")
lines+=['','以下反事实使用真实标签，仅用于定位瓶颈，不能部署。表中是全部有效采样点 RMSE。','','| 反事实诊断 | 415 MHz | 5.775 GHz |','|---|---:|---:|']
for label,key in [('只改正确方向，保持模型原幅度','correct_sign_only'),('给出正确绝对幅度，保持模型原方向','correct_magnitude_only'),('把这4%左右点全部修正准确','perfect_tail_correction')]:
 vals=[z['truth_assisted_diagnostics'][key] for z in d];lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} |')
lines+=['','只修方向仍受过小幅度限制；直接放大错误方向会恶化 RMSE。即便完全修复这批点，其余像素的误差也仍高于双频 1 dB 目标，所以它是首要问题，但不是全部问题。','',
'## 3. 能否学会训练数据：容量诊断','',
'固定旧169表征、全部 FIT 大误差点。使用深度12、3000棵树的高容量回归器，固定训练预算，无早停，也不用 monitor 选择模型。低频169,806点，高频182,138点。','',
'| 容量诊断 | 415 MHz | 5.775 GHz |','|---|---:|---:|']
for label,part,key,pct in [('训练大误差点方向准确率','fit','sign_accuracy',True),('留出大误差点方向准确率','monitor','sign_accuracy',True),('训练大误差点普通RMSE','fit','tail_RMSE',False),('留出大误差点普通RMSE','monitor','tail_RMSE',False),('训练时按原采样权重计算的RMSE','fit','weighted_tail_RMSE',False)]:
 vals=[capacity[str(b)][part][key] for b in range(2)];f=lambda x:f'{x*100:.2f}%' if pct else f'{x:.6f} dB';lines.append(f'| {label} | {f(vals[0])} | {f(vals[1])} |')
lines+=['','方向可以被记住，但不能迁移到新地形。这排除了“方向连训练点都学不会”的解释，不等于证明所有架构都无效。普通训练 RMSE 与加权训练 RMSE 不同；不能把加权数字包装成双频已拟合到1 dB。','',
'## 4. 当前压缩表征是否遗漏了明显可用的空间信息','',
'增加原始接收点33×33地形、TX附近17×17地形，以及TX–RX方向129×9地形带。保留 float32；另提供去除局部平面后的厘米级高差。三个对照使用相同的1,907,908参数网络、初始化、抽样索引和条件大误差 MSE+方向分类损失。原始视图仅由H/TX构造。','',
'控制组分别为旧169输入、旧169+正确原始视图、旧169+跨地形打乱视图。第一轮发现GPU RNG在H100/A800之间产生不同索引，已保留初轮为初步结果，并用统一CPU预生成索引重跑全部三组。以下仅报告修正后的对照。','',
'| 同一真实大误差子集 RMSE | 415 MHz | 5.775 GHz |','|---|---:|---:|']
for mode,label in [('old','旧表征'),('raw','加入正确原始视图'),('shuffled','加入打乱视图')]:
 vals=[z[mode]['tail_RMSE'] for z in v];lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} |')
lines+=['','正确视图没有比旧表征更准。若只在真实大误差点应用输出，正确视图相对旧表征的整体RMSE收益为 -0.000527 / -0.000674 dB，地形配对95%区间均跨零。这否定了本次原始邻域/地形带表示作为明显提分方案的价值；不能推导为“全部原始H/TX没有信息”。','',
'## 5. 哪些点需要大修正：直接控制识别位置','',
'先在真实大误差训练点上训练条件残差回归器。用旧169输入+全图潜在特征的条件模型，如果事先知道大误差位置，整体采样RMSE可到1.924569 / 2.211563 dB。它仍是诊断分数。','',
'为隔离位置识别，另固定旧169条件回归器，比较真实大误差位置与冻结C118模型预测的位置概率。普通点幅度使用FIT均值，没有搜索融合权重或阈值。','',
'| 同一采样点，固定旧169条件回归器 | 415 MHz | 5.775 GHz |','|---|---:|---:|']
for label,key in [('C82','C82_RMSE'),('使用真实位置，仅诊断','true_membership_RMSE'),('使用预测位置，可部署规则','actual_RMSE')]:
 vals=[z[key] for z in g];lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} |')
lines+=['','在与真实大误差比例相同的预测高风险点数下，位置精确率只有43.79% / 43.91%。这次两阶段规则的收益被位置识别错误和误修正抵消。该控制说明识别位置是关键瓶颈；并不证明所用冻结门控已经最优。','',
'## 6. 针对性修改 C126：完整地形网络的状态与幅度监督','',
'把C123完整原生地形轴向注意力网络改为双频三状态输出：普通、需增加路径损耗、需减少路径损耗；每个状态同时预测条件幅度。推理仅由H/TX特征预测状态概率，再计算期望修正。真实状态从不作为推理输入。','',
'两组都是15,216,076参数，同一初始化、完整FIT2815图/1984地形、batch8、8epoch、CPU固定图顺序、AdamW与选择规则。对照只优化全部有效像素MSE；新版本额外加入类别平衡的状态分类与条件幅度损失。只在FIT估计类别先验。','',
'主结果用原始TRAIN标签与原始有效像素掩码，以NumPy float64独立复算。完整monitor共628图、128地形，有效像素40,864,973 / 37,837,148。此口径不同于前面的2048点/图采样结果。','',
'| 完整有效像素 RMSE | 415 MHz | 5.775 GHz |','|---|---:|---:|']
for mode,label in [('C82','C82'),('risk','同架构，只优化总体MSE'),('state','C126，加入状态与条件幅度监督')]:
 vals=s[mode]['RMSE'];lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} |')
lines += ['',f"新版本相对C82仅改善 {s['state']['gain_vs_C82'][0]:.6f} / {s['state']['gain_vs_C82'][1]:.6f} dB。相对同架构对照，低频变化 {score['state_gain_vs_risk'][0]:.6f} dB，高频改善 {score['state_gain_vs_risk'][1]:.6f} dB。高频微小收益的配对区间为正，但远低于0.20/0.30 dB晋级门槛。",'',
f"在真实大误差点上，预测的大误差概率平均从对照的 {risk['monitor']['mean_tail_probability_on_true_tail'][0]*100:.2f}% / {risk['monitor']['mean_tail_probability_on_true_tail'][1]*100:.2f}% 提高到 {state['monitor']['mean_tail_probability_on_true_tail'][0]*100:.2f}% / {state['monitor']['mean_tail_probability_on_true_tail'][1]*100:.2f}%。若用真实三状态选择条件幅度头，诊断RMSE为 {state['monitor']['true_class_oracle_rmse'][0]:.6f} / {state['monitor']['true_class_oracle_rmse'][1]:.6f} dB。辅助子任务确实在学习，但未转化为足够精确的点级判别和实际分数。",'',
'## 7. 能下什么结论','',
'- 已确认：55%不是信息上限。训练目标、状态不确定性和修正幅度都会改变这个数。','- 已确认：所测强模型可以记住训练点方向，主要失败发生在新地形上的状态判别与迁移。','- 已确认：本轮原始邻域/地形带表示，以及完整地形三状态监督，都没有产生实际的大幅RMSE收益。','- 仍未确认：发布的原始地形/TX是否缺少决定这些米级结构的物理信息。模型失败不能单独证明输入信息缺失，也不能宣布1 dB是不可达上限。','',
'本轮没有达到扩训门槛。未进入484地形确认，未用官方VAL做选择或重新训练，也没有网站提交。训练与评估进程结束后检查并释放本轮占用。','',
'## 复现入口','',
'- C125：audit/c125_tail/PLAN.md、protocol.json、diagnosis.json、capacity_verified.json、neural_comparison.json、frozen_gate_probe.json。','- C125修正后的神经对照：runs/c125_nn_old_fixed、runs/c125_nn_raw_fixed、runs/c125_nn_shuffled_fixed。','- C126：audit/c126_statefield/PLAN.md、protocol.json、score.json；runs/c126_risk、runs/c126_state。','- 源代码：src/c125_conditional_tree.py、src/c125_net.py、src/c125_neural.py、src/c125_capacity.py、src/c126_statefield.py、src/c126_train.py。','- 独立计分：scripts/c125_compare.py、scripts/c126_score.py。','']
report=R/'results/C125-C126_大误差方向与状态判别对照_20261108.md';report.write_text('\n'.join(lines))
summary=dict(C125_diagnosis=diag,capacity=capacity,view_control=view,frozen_gate=gate,C126=score,decision='No promotion or expanded training; goal not achieved',official_val_used=False)
(A/'summary.json').write_text(json.dumps(summary,indent=2));print(str(report));print(hashlib.sha256(report.read_bytes()).hexdigest())
