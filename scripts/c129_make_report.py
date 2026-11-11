"""Consolidate the finite C129 diagnosis after all registered controls finish."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c129_fitgap'
def read(p):return json.loads((R/p).read_text())
def fmt(z):return ' / '.join(f'{v:.6f}' for v in z)
v=read('audit/c129_fitgap/capacity_verified.json')
assert set(v['arms'])=={'n16','n64','w64'}
p=read('audit/c129_fitgap/capacity_protocol.json')
probe=read('audit/c129_fitgap/capacity_probe.json')
heads=read('audit/c129_fitgap/native_heads_verified.json')
orig=np.array(heads['original_fullFIT_RMSE']);bound=np.array(heads['per_map_unregularized_all_native_FIT']['RMSE'])
head_pct=(1-(bound/orig)**2)*100
norm={z:read(f'runs/c129_norm_{z}/result.json') for z in ['batch','global']}
readout=read('audit/c129_fitgap/readout_verified.json')
summary=dict(status='diagnosis_and_controls_complete_no_promotion',official_val_used=False,
 official_C82_RMSE=[2.246377454774535,2.5791498899052434],
 monitor_C82_RMSE=[2.146390507349266,2.41801448492553],
 scope=dict(FIT_maps=2815,FIT_terrains=1984,monitor_maps=628,monitor_terrains=128,full_official_TRAIN_maps=25720),
 exposure=read('audit/c129_fitgap/exposure_audit.json'),
 normalization=norm,norm_verified=read('audit/c129_fitgap/norm_verified.json'),readout=readout,native_heads=heads,capacity=v['arms'],shared_anchor_probe=probe['cases'],
 decisions=dict(normalization_repair='implemented_and_verified; no promotion in same-parent four-epoch control',
 wider_model='improves training fit; fails same128monitor; no fullFIT or fullTRAIN expansion',
 conditional_readout='even full-label permap optimal linear heads recover only small SSE fraction; do not launch hypernetwork on this evidence',
 unresolved=['Whether2815mapfullFIT can fit<1 given muchlarger permap exposure','Howmuch encoderoptimization vs inductivebias vs unavailable rawphysicalstate contributes','Whether a nonlinear decoder can extract useful additional information from these frozenfeatures']),
 per_map_label_oracle_SSE_gain_percent=head_pct.tolist(),
 target_met=False)
(A/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
caprows=[]
for tag,name in [('n16','16图，窄模型'),('n64','64图，窄模型'),('w64','64图，宽模型')]:
 z=v['arms'][tag];f=z['FIT'];m=z['monitor']
 caprows.append(f"| {name} | {f['exposures']} | {f['updates']} | {fmt(f['RMSE'])} | {fmt(m['RMSE'])} |")
anchorrows=[]
for key,name in [('n16_ep250','16图窄模型，250次'),('n16_ep1000','16图窄模型，1000次'),('n64_ep250','64图窄模型，250次'),('w64_ep250','64图宽模型，250次')]:
 z=probe['cases'][key]
 anchorrows.append(f"| {name} | {z['updates']} | {fmt(z['shared16anchor']['rmse'])} | {fmt(z['monitor']['rmse'])} |")
worstrows=[]
for tag,name in [('n16','16图窄模型'),('n64','64图窄模型'),('w64','64图宽模型')]:
 f=v['arms'][tag]['FIT']
 worstrows.append(f"| {name} | {fmt(f['worst_map_RMSE'])} | {f['maps_both_below1']} / {f['maps']} | {fmt(f['tail_RMSE'])} |")
report=f"""# C129：16图拟合、开发训练集与留出泛化的分开诊断

日期：2026-11-08。所有新增源码、计算和原始证据都在服务器 CC。原始 data 仅只读使用。

## 直接结论

本轮没有得到大幅提分版本。修正训练归一化、增加单图训练次数、扩宽同一模型，都没有通过同一128地形留出集的双频各0.20dB门槛。没有扩训，没有使用官方VAL选型，也没有网站提交。正式C82仍为2.246377 / 2.579150dB，双频1dB目标未完成。

“16图能拟合”和“大集学不好”包含两个不同问题。小集训练内低误差，部分依赖大量重复训练；跨地形提分则要求学到可迁移的细尺度修正。增加容量已经改善前者，未解决后者。不能再用16图成功断言完整训练集容量足够，也不能把这些结果当作原始地形/TX信息不足的证明。

## 数据、分数和隔离

- 本轮完整开发FIT是2815张图、1984块地形；不是全部25720张官方TRAIN。
- 固定monitor为原128块地形、628张图，评分使用全部256×256原生有效像素。该集合历次复用，且C82校准见过其中80块地形；不称新盲测。
- 各频段分别累计有效像素SSE和数量，再开方。表中顺序为415MHz / 5.775GHz，单位dB。训练归一化对照涉及包含权重；表中主要分数是普通有效像素RMSE，未混用。
- 掩码只用于loss和计分，不作为输入。未读取或复用CC外另一AI的代码、权重或特征。
- 只使用授权空闲1/3/4卡的完整UUID和run_gpu.sh。没有修改驱动或终止其他人的进程。

## 一、原比较混合了训练强度和容量

上一轮16图诊断每张图重复1700次，共1700次参数更新。2815图主试验最多每张图24次，共4395次更新。前者每图训练次数约为后者71倍；后者总更新数更多。因此，不能只按“训练步数”或只按“图数”解释差距。

本轮用嵌套16图和64图：4/16块多TX地形，每块4个TX。16图是64图子集。相同预训练来源、C82初始预测、普通MSE、batch16、AdamW学习率0.0004和weight decay0.01。两组64图使用同一样本顺序。

窄模型1,529,860参数；宽模型5,035,684参数。扩宽保留深度和注意力head_dim。函数保持的通道复制在非零父模型上重放最大误差4.77e-6；之后两组都施加相对1e-3微扰，初始零头仍精确输出C82。宽度也可能改变优化条件，所以这是同模型族的宽度干预，不是“纯表达容量”的数学证明。

所有组最多1700次/图；最早100次之后，训练汇总双频都低于0.95dB才提前结束。停止不读取monitor分数。结束后独立读取原始标签和掩码，以NumPy FP64复算。

| 对照 | 每图训练次数 | 参数更新数 | 训练汇总RMSE | 相同monitor RMSE |
|---|---:|---:|---:|---:|
{chr(10).join(caprows)}
| 固定C82 | — | — | 各训练子集不同，不混报 | 2.146391 / 2.418014 |

不同组终点由FIT停止规则决定，不能用不同终点直接归因宽度。下面固定相同次数，并只比较共同16图锚点：

| 对照 | 参数更新数 | 共同16图RMSE | 相同monitor RMSE |
|---|---:|---:|---:|
{chr(10).join(anchorrows)}

250次/图时，64图窄/宽模型在全部64图上分别为1.384106/1.522479和1.205335/1.346634。宽模型降低训练SSE约24.2%/21.8%，但monitor仅改善0.007980/0.020827dB，仍明显差于C82。

相同250次/图下，64图模型在共同16图上没有劣于16图模型。相同1000更新下，16图模型因为每图看到1000次而拟合更低，64图只有250次。数据量、单图重复量和更新次数确实不能混为一谈。此结果只覆盖16→64图，不能外推保证2815图可在同预算达到1dB。

训练汇总低于1dB也不代表每图或大误差区达标：

| 对照 | 最差图RMSE | 双频均<1dB的图数 | C82绝对误差>=5dB区域RMSE |
|---|---:|---:|---:|
{chr(10).join(worstrows)}

## 二、找到并修复归一化偏差，但它没有带来提分

原训练使用每个batch内的“加权SSE/加权像素数”。在当前成组顺序中，这不等于整集加权目标：多TX样本在预定目标中的有效权重占比应为89.22%/89.34%，实际batch平均目标约79.79%/79.80%。

新实现使用每频段“整集加权有效像素数/本轮batch数”作为固定分母。固定预测的FP64审查中，所有batch损失的平均值与完整目标相符，数值误差约8e-15。该修复已用于C129新训练代码，旧实验源码和权重保持原样。

从同一个C127第4轮父权重出发，原分母与修复分母做同顺序、同优化器续训。两组都跑到事先规则要求的第4轮，最佳仍是第0轮父模型：2.142057/2.414297dB。修复是有依据的实现改动，但在此续训协议内不是大幅提分原因。不能据此排除所有学习率、优化器或更长训练的可能性。

## 三、输出层已经接近最优；现有冻结特征难以线性读出剩余误差

冻结当前C127普通残差父模型，取48维TX分支特征和24维地形分支特征，总计72维。用原C101每图1536个FIT点求加权线性最小二乘。无正则最优解与原头的加权训练RMSE分别为2.194822/2.454840与2.196107/2.457630，几乎没有差距。

完整2815图的独立原始标签计分：

| 输出规则 | 完整FIT普通RMSE | 原128monitor RMSE |
|---|---:|---:|
| 原父模型输出头 | {fmt(readout['parentFIT']['RMSE'])} | 2.142057 / 2.414297 |
| 冻结特征的闭式输出头 | {fmt(readout['closedFIT']['RMSE'])} | {fmt(readout['closedMonitor']['RMSE'])} |

这只排除了“最后一个共享线性头没优化好”作为主要原因，不等于整个编码器达到最优。

进一步允许每块地形、甚至每张图使用自己的标签拟合不同的线性头。先前768点/图的检查出现过拟合，故仅做一次预先固定的全像素复核：棋盘一半像素32768点拟合，另一半检查；固定标准化ridge=1e-3，没有按结果调参。父编码器曾训练全部FIT像素，因此这是输出层诊断，不是新数据泛化实验。

| 输出规则 | 同图另一半像素RMSE |
|---|---:|
| 原输出头 | 2.175596 / 2.457107 |
| 每地形独立输出头，使用FIT标签 | 2.158310 / 2.427776 |
| 每图独立输出头，使用FIT标签 | 2.152538 / 2.420030 |

再给每张图全部原生有效标签，直接求无正则最优线性头，连计分点的标签都可用于拟合。此有利诊断仍为{fmt(bound)}，只减少约{head_pct[0]:.2f}%/{head_pct[1]:.2f}%的训练SSE。CPU独立重解已通过，原头双半像素统计与单独原始标签计分一致。

所以，更换共享输出头、或只让每张图选择不同线性输出头，都缺少明显提分依据。没有因此启动条件超网络。这里的标签辅助结果不能部署，也不是地形/TX原始输入的信息下限；更复杂的非线性读出也未被这个对照穷尽。

## 四、这次能下的判断与不能下的判断

| 因素 | 当前证据支持 | 仍不能断言 |
|---|---|---|
| 容量 | 扩宽明显加快64图拟合；窄模型不是完全无法学习残差 | 2815图容量足够，或只加参数就能泛化 |
| 优化 | 原比较单图训练次数严重不等；存在归一化偏差并已修复；最终线性头优化余量很小 | 编码器已经全局最优，或继续训练一定无效 |
| 表征 | 当前72维表示不支持大幅线性纠错；拟合增强没有迁移到相同新地形 | 原始H/TX必然缺信息，或双频1dB不可达 |

核心矛盾是：模型能够把特定地形上的细尺度误差学进去，但当前训练得到的表示不能在新地形上稳定给出修正方向和幅度。更多参数和更多重复训练目前增强的是训练内拟合，不是参赛得分。

本轮采取的针对性修改是准确的整集损失归一化和同族扩宽；两项都已经接受同一monitor检验，并被否决为扩训候选。尚未找到有证据支持的新表征方案。不得把负结果包装成“已解决”，也不应据此继续堆输出头、调融合或长训。

## 复现与证据

- 开始前问题与晋级门槛：audit/c129_fitgap/PLAN.md、capacity_protocol.json。
- 曝光量和权重：exposure_audit.json、weight_audit.json、norm_smoke.json。
- 归一化对照：src/c129_norm.py；runs/c129_norm_batch、runs/c129_norm_global。
- 冻结读出：src/c129_readout.py；readout.json、readout_verified.json、unregularized_head_bound.json。
- 容量对照：src/c129_capacity_model.py、src/c129_capacity.py、src/c129_capacity_probe.py；runs/c129_capacity_*、capacity_verified.json、capacity_probe.json。
- 全像素条件读出：src/c129_native_heads.py；conditional_native_heads.json、native_heads_verified.json。
- 独立计分：scripts/c129_verify_norm.py、scripts/c129_verify_readout.py、scripts/c129_verify_capacity.py、scripts/c129_verify_native_heads.py。
- 机器汇总：audit/c129_fitgap/summary.json。资源终态另见final_resources.json。

原484地形确认和全部官方TRAIN扩训均未触发。正式官方分数未变化。
"""
(R/'results/C129_小集与完整开发训练集拟合差距_20261108.md').write_text(report)
print('wrote report and summary')
