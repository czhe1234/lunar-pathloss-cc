from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1];A=R/'audit/c127_joint'
def read(p):return json.loads((R/p).read_text())
score=read('audit/c127_joint/score.json');capacity=read('audit/c127_joint/capacity_verified.json');info=read('audit/c127_joint/tx_information.json');pose=read('audit/c128_pose/result.json')
def pair(v,d=6):return f'{v[0]:.{d}f} / {v[1]:.{d}f}'
arms=['residual_independent','residual_joint','direct_independent','direct_joint']
labels={'C82':'C82','residual_independent':'残差 + 普通监督','residual_joint':'残差 + 联合监督','direct_independent':'直接预测 + 普通监督','direct_joint':'直接预测 + 联合监督'}
models=score['models']
best=min(arms,key=lambda n:sum(models[n]['rmse']))
lines=['# C127–C128：多 TX、双频组织与直接预测对照',
'',
'结论：本轮没有得到大幅提分版本。已实现并完成四组训练、独立计分，以及跨 TX 对应关系的无训练诊断。所有分支均未达到原留出集双频各0.20 dB的门槛，不进入确认集、不扩训、不替换正式版本。',
'',
f'同一128地形/628图完整有效像素，C82为 {pair(models["C82"]["rmse"])} dB；本轮按双频平均RMSE选出的候选为“{labels[best]}”，{pair(models[best]["rmse"])} dB，仅改善 {pair(models[best]["gain_vs_C82"])} dB。',
'',
'官方固定VAL本轮未用于训练或选择，也未重新评分。正式C82仍为2.246377 / 2.579150 dB。双频不高于1 dB目标未完成。',
'',
'## 1. 实际修改',
'',
'网络明确分为地形分支 S(H,RX) 和 TX 条件分支 T(H,TX,RX)，共同输出双频路径损耗。地形分支只读地形派生量；多个TX对同一地形的编码只为节省计算而复用。没有图ID嵌入，没有真实功率或mask作为推理输入。',
'',
'共同成分的监督只是四个已选TX的样本均值，含TX采样波动，不是已知的物理场景状态。联合监督用同一FIT地形四个TX的真实目标构造共同均值和各TX偏离，分别监督两个分支；另监督共同有效像素的双频差异。真实组均值仅是训练目标。普通监督对照保留同一网络、同一组批次，只使用逐点MSE。因此本次比较不是单纯改批次顺序，也不是把MSE作等价正交重写。',
'',
'四组交叉两个因素：直接预测/修正C82，以及普通/联合监督。直接预测使用H/TX计算的物理先验和已有C99权重，不读取C82输出；残差路线以C82为参考。两条路线的起始预测不同，不能把直接与残差比较当作同一初始化下的纯因果对照。每条路线内的普通/联合对照使用完全相同初始化。',
'',
'TX分支继承CC中已有C99骨干；地形分支新增，总参数1,529,860。所有新更新均用原2815张FIT图/1984地形。父权重此前使用过更多FIT数据；两组共享同一父权重。C99骨干训练排除了原监测地形。本轮新训练损失没有使用监测标签。',
'',
'历史边界：这个监测面板已被多轮用于选型，C82校准曾用其中80块地形。它是固定开发比较面板，不是全新盲测。微小收益的区间不能当作未选择偏差下的最终泛化保证。',
'',
'## 2. 对照结果',
'',
'单位dB；低频415MHz，高频5.775GHz。原始TRAIN标签和原始有效像素mask经NumPy float64独立复算，低/高频有效像素40,864,973 / 37,837,148。',
'',
'| 方法 | 低频RMSE | 高频RMSE | 最佳/结束轮 |',
'|---|---:|---:|---:|']
for name in ['C82']+arms:
 m=models[name];ep='—' if name=='C82' else f'{m["best_epoch"]}/{m["last_epoch"]}'
 lines.append(f'| {labels[name]} | {m["rmse"][0]:.6f} | {m["rmse"][1]:.6f} | {ep} |')
lines+=['',
'两条路线均没有明显改善。下表是联合监督相对同路线普通监督的收益，正值为改善：','',
'| 路线 | 低频收益 | 高频收益 |','|---|---:|---:|']
for route,label in [('residual','残差'),('direct','直接预测')]:
 g=score['joint_effect'][route]['joint_gain_vs_independent'];lines.append(f'| {label} | {g[0]:.6f} | {g[1]:.6f} |')
lines+=['','逐地形配对95%区间保存在 audit/c127_joint/score.json。即使某项区间为正，千分之几dB也没有达到实际提分门槛。','',
'## 3. 大误差点是否真正改善','',
'大误差子集固定为C82误差绝对值至少5dB；真实标签仅用于评价。','',
'| 方法 | 尾部RMSE低/高频 | 尾部修正方向准确率低/高频 |','|---|---|---|']
for name in arms:
 m=models[name];lines.append(f'| {labels[name]} | {pair(m["tail_rmse"])} | {pair([100*z for z in m["tail_sign_accuracy"]],2)} % |')
lines+=['','方向、位置和功率没有形成能显著改善整体RMSE的联合结果。分数来自部署可用的H/TX输入；没有把已知真实大误差位置的诊断分数混入此表。','',
'## 4. 小集能够优化，但不代表完整训练集已充分拟合','',
f'同一新网络与联合损失，在4块FIT地形/16张图上，{capacity["step"]}步达到汇总 {pair(capacity["RMSE"])} dB。最差单图为 {pair(capacity["per_band_worst"])} dB；尾部RMSE为 {pair(capacity["tail_RMSE"])} dB。并非每张图或尾部都低于1dB。',
'',
'这是FIT容量诊断，不是留出或官方分数。它只排除新增分解完全无法优化，不能证明完整数据容量足够。主实验的固定192张FIT面板仍未充分拟合；因此不能将全部失败直接归为“泛化”或“输入缺信息”。',
'',
'## 5. 共享关系的直接证据','',
'原2815张开发FIT图只有277块地形含四个TX，其余1707块各只有一个TX。多TX组内两两水平距离中位数约124.13个1m网格步长。因此“同一地形”不等于“相近传播条件”。',
'',
'先用另外三个TX的真实残差来预测当前TX，属于诊断供体，不可部署：',
'',
f'- 同一可比较点上的C82：{pair(info["reference_RMSE"])} dB。',
f'- 减去其他TX真实残差均值：{pair(info["subtract_other3_mean_RMSE"])} dB，反而更差。',
f'- 该均值在大误差点上的方向准确率：{pair([100*z for z in info["tail_otherTX_direction_accuracy"]],2)}%。',
f'- 若把当前TX自己的真实残差也混进四TX均值，诊断分数会变成 {pair(info["subtract_own_inclusive4_mean_RMSE"])} dB。这含当前标签，不是跨TX可预测性证据。',
'',
'再做C128：将供体整数平移到相同RX−TX相对坐标。原位置和相对位置使用完全相同的供体可用性、query点和标签掩码；未插值，未修复地形网格，不涉及C95。因覆盖条件不同，不能与上一组RMSE直接比较。',
'',
f'- 该共同点集C82：{pair(pose["reference"])} dB。',
f'- 原RX位置供体修正：{pair(pose["same_RX_donor"])} dB。',
f'- TX相对位置供体修正：{pair(pose["TX_relative_donor"])} dB。',
f'- 相对位置供体的尾部方向准确率：{pair([100*z for z in pose["direction_TX_relative"]],2)}%。',
'',
'即使给出额外真实供体标签，简单原位或平移共享也没有找到提分依据。它否定的是这两种共享规则，不能否定所有跨TX建模，也不能证明原始H/TX不充分。没有据此启动TX中心架构训练。',
'',
'## 6. 精度、资源与决策',
'',
'首次烟雾检查发现AMP下批次形状改变会影响舍入。分开检查后，AMP预测差RMS0.011744dB、最大0.17524dB；高精度检查RMS0.00007867dB，未发现跨样本功能依赖。直接模型更换为FP32读出后，与原C99的AMP重放差RMS0.007790dB。所有正式对照使用同一实现和批次序列；微小收益不作重大改进。',
'',
'主对照仅使用空闲的H100实体1/3；A800实体4用于烟雾和小集诊断。按UUID限定设备，未触碰其他任务。模型、脚本、数据缓存与报告均在CC。无确认集评分、无全TRAIN扩训、无官方VAL选择、无网站提交。',
'',
'当前决策：关闭本次固定接收位置共同成分分支；不推广简单TX平移共享。没有形成满足用户要求的新版本。输入充分性和完整数据的可学习性仍未被本轮完全区分。',
'',
'## 复现',
'',
'- audit/c127_joint/PLAN.md、protocol.json、score.json、capacity_verified.json、tx_information.json。',
'- audit/c128_pose/PLAN.md、result.json、terrain_stats.npz。',
'- src/c127_joint.py、src/c127_train.py、src/c127_capacity.py。',
'- scripts/c127_score.py、scripts/c127_tx_information.py、scripts/c128_pose.py。',
'- runs/c127_{residual,direct}_{independent,joint} 下的配置、历史、权重和完整monitor预测。',
'- logs/c127_*.log、logs/c128_pose.log。']
text='\n'.join(lines)+'\n'
path=R/'results/C127-C128_多TX双频联合与直接预测对照_20261108.md';path.write_text(text)
summary=dict(status='complete_no_promotion',best_monitor_model=best,score=score,capacity=capacity,TX_information=info,relative_TX_alignment=pose,
 decision='No monitor gate pass; do not confirm or expand. Official target not reached.',
 report=str(path.relative_to(R)),report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),official_val_used=False)
(A/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
manifest=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='C127fourarm/C128untraineddiagnostic',commands=[
 './run_gpu.sh 4 src/c127_train.py --smoke',
 './run_gpu.sh 1 src/c127_train.py --route residual --mode independent',
 './run_gpu.sh 3 src/c127_train.py --route residual --mode joint',
 './run_gpu.sh 1 src/c127_train.py --route direct --mode independent',
 './run_gpu.sh 3 src/c127_train.py --route direct --mode joint',
 './run_gpu.sh 4 src/c127_capacity.py',
 'CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/c127_score.py',
 'CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/c127_tx_information.py',
 'CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/c128_pose.py'],
 arms={name:read('runs/c127_'+name+'/config.json') for name in arms},result_files={name:'runs/c127_'+name+'/result.json' for name in arms},summary='audit/c127_joint/summary.json',
 source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'src/c127_joint.py',R/'src/c127_train.py',R/'src/c127_capacity.py',R/'scripts/c127_score.py',R/'scripts/c127_tx_information.py',R/'scripts/c128_pose.py']})
(A/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(report=str(path),best=best,rmse=models[best]['rmse'],gain=models[best]['gain_vs_C82'],sha256=summary['report_sha256']),ensure_ascii=False))
