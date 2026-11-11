from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c130_projective'
def load(n):return json.loads((A/n).read_text())
def fmt(x,d=6):return ' / '.join(f'{a:.{d}f}' for a in x)
probe=load('probe_verified.json');projection=load('projection_result.json');rank=load('tx_rank_result.json');branch=load('branch_audit.json');sub=load('subgroups.json')
assert not probe['gate_passed']
summary=dict(status='completed_no_promotion',official_C82_RMSE=[2.246377454774535,2.5791498899052434],official_val_used=False,target_met=False,
 projection=projection,rank=rank,probe=probe,branch=branch,subgroups=sub,verification=load('diagnostics_verified.json'),resources=load('final_resources.json'),
 decision='Do not train projection-transfer, low-rank shared-TX, or expand bottleneck-node candidate; no qualified new information route found',
 unresolved='RawH/TX sufficiency vs unavailablephysicalstate vs a better learnedrepresentation is not settled; these controls do not establish an irreducible error floor.')
(A/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for mode,name in [('old','旧169维输入'),('nodes','旧输入＋几何瓶颈节点'),('shuffled','旧输入＋打乱瓶颈节点')]:
 z=probe['results']['monitor'][mode]
 rows.append(f"| {name} | {fmt(z['tail_RMSE'])} | {fmt(z['truth_gated_all_RMSE'])} |")
report=f"""# C130：地形与TX细尺度信息的有效利用

2026-11-08。源文件、数据分析和GPU计算都在 transformer:/home/chenzhe/work/lunar-pathloss-rm-challenge/CC。此报告的本地文件是查看副本。

## 结果与决定

本轮没有找到已经证明能大幅提分的信息缺口。完成两项数据关系诊断，并实现了新的“几何瓶颈检索＋局部CNN＋跨节点注意力”表征方法。它未通过同一留出集上的预检，不进入全点训练、484地形确认或官方TRAIN扩训。没有网站提交。

正式C82官方验证仍为2.246377 / 2.579150dB，双频1dB目标未完成。以下数值均按415MHz / 5.775GHz排列。

## 1. 三维出射方向不能直接迁移剩余误差

277块开发FIT多TX地形、每块4个TX。固定每图原C101前512个查询点。只用发布地形H和TX，把查询接收点匹配到另一TX下三维出射方向最近的接收点。对比固定RX位置、简单TX平移和三维方向对应。旧方法只作为相同支持域的对照，没有重启旧候选。

对应函数不读标签或mask；其他TX的真实残差仅用于有利诊断。所有比较使用共同有效支持。三维方向近匹配的容差事先固定为角度误差乘距离不超过0.5m。

近匹配中，三维方向对应残差相关为-0.002206 / -0.000416，地形重采样95%区间为[-0.004830,0.000327] / [-0.003478,0.002742]。直接借用对应位置的真实残差，RMSE从1.851160/1.897253恶化到2.337088/2.468095。没有据此训练投影坐标网络。

该对应使用发布栅格、TX相对地面3m与RX相对地面1m，不代表找回了真实网格或实际传播路径。没有重启C95。

## 2. 多TX细结构不适合直接共享少量空间模板

原始TRAIN中392块地形各有51个TX。近期开发FIT对其中277块各取4个TX。本轮固定种子选32块FIT地形，读取每块51张原始功率图。每次留出一个TX，只用另50个TX建立空间基底，均值也排除查询TX。

去自由空间损耗后，再去掉sigma=4像素的平滑部分，研究细尺度变化。即使允许查询图的真值参与求最佳组合，使用全部49维中心化基底后仍剩2.562588 / 2.603349dB；只用其他TX均值时为2.791482 / 2.941458dB。直接最小二乘与独立Gram矩阵算法在一份真实样本上一致。

这是标签辅助的线性空间子空间诊断，不是任何输入模型的信息下限。共同有效域覆盖完整栅格的87.67% / 46.01%，高频覆盖尤其有限，不能当全域官方分数。该结果不支持立即启动低秩共享模型。

## 3. 新方法读取关键障碍周围的二维地形

代码审查发现：旧C4/C70压缩关键障碍统计；旧C125显式读取TX/RX附近补丁和直线走廊。本轮检验一个不同的信息组织方式：

1. 用发布H/TX构建512点剖面，选归一化净空最不利的主节点，再在左右两段递归选节点。
2. 在三个节点周围，读取沿路径/横向坐标中的17×17地形。采样间隔为1m和4m，保留相对高程及去局部平面后的细节。
3. 用共享局部CNN编码各节点，加入位置、净空、坡度等8维几何量。节点间用注意力交互，再由旧169维条件查询节点信息。

这是候选几何瓶颈，不是已知真实传播路径。平地、单山脊和90度旋转检查通过；首次测试的整数合成地形类型问题已修复，失败日志保留。提取没有读取功率或mask。

三组参数完全相同，均为790548参数，使用同一初始权重和抽样顺序：旧输入、新节点、打乱节点。打乱节点来自其他地形，切断当前H/TX与新增视图的对应。固定最大4000更新；三组均第1000步最佳，第2000步按预定规则停止。

## 4. 同一留出集没有明显收益

沿用原128块地形的C101固定点。FIT使用265605个、monitor使用75213个真实大误差点的并集，分别按频段有效性和|C82残差|>=5dB监督。

本阶段刻意给模型“已知哪些点需要修正”的有利条件。表中“真值门控全点RMSE”只在真实大误差位置应用预测修正，其余位置保留C82。它不能部署，也不能当作参赛成绩。

| 条件对照 | 大误差点RMSE | 真值门控全点RMSE |
|---|---:|---:|
{chr(10).join(rows)}

新节点相对旧输入的改善为0.000297 / -0.001215dB，配对地形95%区间[-0.001647,0.002206] / [-0.004544,0.002004]。相对打乱输入的改善为0.000463 / 0.000053dB，区间也都跨零。未达到事先固定的双频各0.10dB条件预检门槛，更没有证明可部署全点模型满足0.20dB门槛。

原始TRAIN标签、按位读取的原始mask、NumPy FP64独立计分已核验。该点面板C82为2.140018/2.411312；不能与原生整图monitor的2.146391/2.418014混用。

## 5. 分支学起来了，但没有形成可迁移的修正规则

固定最后第2000步权重，把正确节点换成打乱节点：

| 评价范围 | 正确节点的大误差点RMSE | 打乱节点的大误差点RMSE |
|---|---:|---:|
| FIT | 8.140512 / 8.891280 | 8.771722 / 9.570320 |
| monitor | 8.969671 / 9.608853 | 8.970531 / 9.646914 |

正确对应在训练内有0.631210/0.679039dB作用，在留出上仅0.000860/0.038062dB。这说明分支确实利用了新输入；不能把失败简单归为代码未接通。但输入置换可能造成分布变化，结果也不证明原始地形信息必然无效。

还检查了固定几何分区：主节点净空为正时，留出尾部仅改善0.005911/0.002942dB。约89.16%/87.56%的尾部点，主节点位于旧端点补丁范围之外；这部分也只有0.001108/-0.004697dB。没有发现被整体均值掩盖的大幅双频收益。分区没有用于调门槛或构造新融合。

## 结论边界

当前证据支持：剩余细结构不能靠简单方向对应或少量跨TX空间模板解决；新增关键障碍二维形状被模型利用，但本轮学习到的关系主要改善训练内拟合，未带来跨地形收益。

尚未证明：原始H/TX缺少必需信息、任何更强表征都无效、双频1dB不可达。本轮没有确定一个可扩训的新方法。没有因负结果继续堆网络宽度、输出头或融合权重。

另核对了CC保存的Sionna1.1.0源码及官方技术说明：RadioMap累加不同路径的功率，未保留路径间相干交叉项。因此没有把这些细结构直接认作相干干涉，也没有以此为由增加相位网络。[Sionna Radio Map Solver](https://nvlabs.github.io/sionna/rt/tech-report/S4.html)

## 文件与资源

- 问题、顺序和门槛：audit/c130_projective/PLAN.md、probe_protocol.json。
- 数据关系诊断：scripts/c130_projection_audit.py、scripts/c130_tx_rank.py。
- 新表征及对照：src/c130_nodes.py、src/c130_probe_model.py、src/c130_probe_train.py。
- 权重、训练轨迹和预测：runs/c130_nodes_old、runs/c130_nodes_nodes、runs/c130_nodes_shuffled。
- 独立计分与诊断核验：scripts/c130_score_probe.py、scripts/c130_verify_diagnostics.py。
- 固定权重置换：src/c130_branch_audit.py；branch_audit.json。
- 机器汇总：audit/c130_projective/summary.json。

本轮只使用授权空闲1/3/4完整UUID。所有GPU任务已退出，最终快照均17MiB/0%，无计算进程。未干预其他任务。
"""
(R/'results/C130_地形与TX细尺度表征对照_20261108.md').write_text(report)
print('report and summary written')
