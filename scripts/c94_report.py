"""Consolidate the native-cell diagnostic. No training or parameter selection."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1]
A=R/'audit/c94_measurement';O=R/'results';O.mkdir(exist_ok=True)
b=json.loads((A/'bounds.json').read_text())
s=json.loads((A/'templates_score.json').read_text())
e=b['selected_counterexample']
modes=['linear_power_interpolation','dB_interpolation','weakest_donor_any_convex_interpolation']
counts=[tuple(b['bounds'][str(j)][m]['all_native_contributors_positive']['exceed5'] for j in (0,1)) for m in modes]
old=s['scores']['old_CC'];new=s['scores']['templates_numpy126/sionna_centers_default']
report=f"""# C94 最关键的矛盾

2026-11-03。所有计算在 CC 内完成。仅用 CPU。没有训练、调整融合权重、使用官方 VAL 标签选型或网站提交。

## 结论

当前重建的地形、TX、接收单元和像素映射，不能同时解释原版求解器的功率定义与发布标签。至少一项观测生成假设尚未复原。

这个问题比补充反射和散射功率更靠前。按原版求解器的定义，增加其他路径不能抵消已经记录到的直达功率。因此，解释反例需要改变候选几何、TX 或观测对应，或者找到未被复现的官方生成设置。

“通视”仅指 CC 当前候选几何下的判断。尚未获得原始比赛网格或完整官方导出脚本。本结果不能证明发布数据违反物理规律，也不能证明仅靠现有输入绝对达不到 1 dB。

## 原版求解器的约束

已核对 CC 内固定版本 Sionna RT 1.1.0 的源码 mesh_radio_map.py 中 MeshRadioMap.add。每条射线的场先取模平方，再乘非负的立体角、入射角和单元面积权重，最后通过 scatter_reduce(Add) 在接收单元内累加。

本赛题为单个 iso/V 发射天线单元。对于同一场景、同一发射方向和同一接收单元，总功率不低于直达功率。本结论针对该原版 radio-map 求解器，不能用复电场相消解释其逐路径非相干累加结果。

初始方向来自固定 Fibonacci lattice。固定发射数量、H 和 TX 时，更换后续交互采样的随机种子，不会重新随机化初始直达命中。

## 必须纠正的比较方式

直接比较栅格直达功率和栅格总功率，可能混入插值域差异。

例如原始功率为 1 和 0.001，权重各为 0.5。先平均线性功率再取路损是 {b['jensen_demo']['linear_power_PL']:.4f} dB。直接平均路损是 15 dB。两者相差约 12 dB。

此轮回到每个像素对应的三个候选原始三角单元，保存原版求解器返回的直达值。对于同一供体集合，不论在线性功率域还是 dB 域作非负、权重和为 1 的插值，都必须满足：

    标签总路损 ≤ 候选供体中最大的直达路损

最大的直达路损对应最弱的直达功率。所有有效供体都必须有正直达功率。零直达供体保留为无穷路损，不能裁成有限值后制造反例。

## 独立计数

沿用 C93 事先根据 H/TX 选择的 60 块 TRAIN 留出地形、117 个连续窗口。窗口离 TX 至少 65 m，窗口内候选几何通视比例至少 80%。没有按本轮标签误差筛选窗口。

原版求解器使用 CPU 后端和标称 40 亿初始发射方向。只计算与目标接收单元相关的初始方向。原有 C93 已检查方向搜索范围的扩展。此轮另存每个原始供体值，并重复核对旧名义相位输出。

每频段共 5,683 个有效、候选几何通视的位置。其中 4,575 个位置的全部有效供体均有正直达功率。下表使用同一组 4,575 个位置，计数为标签路损超过相应上界 5 dB。

| 上界定义 | 415 MHz | 5.775 GHz |
|---|---:|---:|
| 固定权重，线性功率插值后取 dB | {counts[0][0]} | {counts[0][1]} |
| 固定权重，直接插值单元 dB | {counts[1][0]} | {counts[1][1]} |
| 最弱直达供体，允许任意凸插值权重 | {counts[2][0]} | {counts[2][1]} |

最后一行排除了只改变这三个供体上的权重或功率/dB 插值域这一解释。它没有排除换供体、换网格、换坐标或其他未复现的生成设置。

## 可复算的双频反例

TRAIN 样本 {e['sample']}，索引行 {e['index']}，接收像素第 {e['rx_row']} 行、第 {e['rx_col']} 列。行列编号从 0 开始。

候选供体编号为 {e['donor_ids']}。权重为 {[round(v,8) for v in e['donor_weights']]}。

| 数值 / dB | 415 MHz | 5.775 GHz |
|---|---:|---:|
| 供体一的直达路损 | {e['direct_PL_native_donors'][0][0]:.4f} | {e['direct_PL_native_donors'][1][0]:.4f} |
| 供体二的直达路损 | {e['direct_PL_native_donors'][0][1]:.4f} | {e['direct_PL_native_donors'][1][1]:.4f} |
| 供体三的直达路损 | {e['direct_PL_native_donors'][0][2]:.4f} | {e['direct_PL_native_donors'][1][2]:.4f} |
| 最弱直达供体允许的最大路损 | {e['bound_weakest_native'][0]:.4f} | {e['bound_weakest_native'][1]:.4f} |
| 发布的总路损标签 | {e['true_PL'][0]:.4f} | {e['true_PL'][1]:.4f} |
| 标签超出上界 | {e['gap_past_weakest'][0]:.4f} | {e['gap_past_weakest'][1]:.4f} |

97 个位置同时满足双频超界至少 5 dB。上例按双频平均超界量最接近中位数选择，用于展示计算，不代表随机抽样误差分布。

## 已发现的模板差异没有解决反例

旧 CC 模板用 NumPy 求接收三角形中心，原版 Sionna 用 DrJit。极小的浮点坐标差会改变近规则中心网格的 Delaunay 三角剖分。

全图 65,536 个像素中，45.6818% 的像素有效供体权重 L1 变化超过 0.01，28.1174% 超过 0.5。这是可复现的实现差异。

已比较当前 NumPy 2.2.6 和隔离安装的 NumPy 1.26.4、三种中心构造、正反顺序、三种输出网格精度，共 36 个新模板和旧模板。隔离安装未修改项目原有虚拟环境。

在所有模板均有保存供体值的 5,012 个有效、候选几何通视位置，旧模板“直达比标签总信号强超过 5 dB”的计数为 {old[0]['direct_alone_5dB_too_strong']} / {old[1]['direct_alone_5dB_too_strong']}。换成原版中心和 NumPy 1.26.4 后为 {new[0]['direct_alone_5dB_too_strong']} / {new[1]['direct_alone_5dB_too_strong']}，没有减少。

此处为固定线性功率插值，允许部分供体直达为零。与上表“最弱供体、全部供体正值”检验的分母和定义不同。

弱点排序 AP 仅由 {old[0]['AP_rank_miss']:.4f} / {old[1]['AP_rank_miss']:.4f} 变为 {new[0]['AP_rank_miss']:.4f} / {new[1]['AP_rank_miss']:.4f}。未形成可靠定位方法，未得到新的官方分数。

## 邻域检查提示与局限

具有完整 1 m 邻域的硬反例中，低频 71 个里 67 个、高频 68 个里 65 个，可以找到直达足够弱的其他供体。其中低频 66 个、高频 63 个邻域有零直达供体。

细小供体对应差别有能力产生大功率变化。但该检查使用标签后放宽可选供体，只是上界，未恢复由 H/TX 独立决定的盲测映射。其余非硬反例也经常有零直达邻居，不能据此自动改弱点。

## 后续决策

先使用具体反例检查原始地形网格、TX 亚像素坐标、接收面构造、像素坐标和插值供体。每个候选都必须从 H/TX 独立确定，在未用于选型的地形上同时解释弱点位置和功率。

当前补充非负反射或散射功率的候选无法修复直达单独已经过强的像素。加强网络训练不能证明观测对应已经解决。换成原版中心坐标这项具体修复已失败，不沿此方向扩训。

尚无证据把全部误差归因于缺失输入，尚未证明双频 1 dB 不可达。

完整逐像素值在 audit/c94_measurement/pixel_bounds.csv。原始单元值在 audit/c93_joint_miss/native0、native1、native2。计算入口为 scripts/c94_bounds.py 和 scripts/c94_template_score.py。本报告和图由 scripts/c94_report.py 生成。

本轮没有新官方分数。C82 仍为 2.246377 / 2.579150 dB，官方综合 2.412764 dB。
"""
(O/'C94最关键矛盾_20261103.md').write_text(report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
cjk=next((p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p),None)
if cjk:plt.rcParams['font.family']=font_manager.FontProperties(fname=cjk).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'figure.dpi':160})
a=np.genfromtxt(A/'pixel_bounds.csv',delimiter=',',names=True,dtype=None,encoding='utf-8-sig')
fig,axes=plt.subplots(1,2,figsize=(11,5.25))
for j,(band,title) in enumerate([('415','415 MHz'),('5775','5.775 GHz')]):
    ax=axes[j];m=(a['all_positive']==1)&(a['valid'+band]==1)
    x=a['worst'+band][m];y=a['Y'+band][m];hard=y-x>5
    lo=np.floor(min(x.min(),y.min())/5)*5;hi=np.ceil(max(x.max(),y.max())/5)*5
    ax.scatter(x[~hard],y[~hard],s=5,alpha=.22,color='#0072B2',rasterized=True,label='其余检查点')
    ax.scatter(x[hard],y[hard],s=14,alpha=.85,color='#D55E00',rasterized=True,label='超出上界 5 dB')
    ax.plot([lo,hi],[lo,hi],color='#333333',lw=1,label='同供体的必要上界')
    ax.plot([lo,hi-5],[lo+5,hi],color='#D55E00',lw=.8,ls='--')
    ax.scatter([e['bound_weakest_native'][j]],[e['true_PL'][j]],marker='*',s=130,facecolor='#F0E442',edgecolor='#111111',lw=.7,zorder=5,label='表中双频反例')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal',xlabel='候选供体中最大的直达路损 / dB',ylabel='发布的总路损标签 / dB',title=title+f"   {hard.sum()} / {m.sum()} 个超界点")
    ax.grid(alpha=.12)
axes[0].legend(loc='lower right',fontsize=8.7,framealpha=.9)
fig.suptitle('当前候选观测对应无法解释部分标签',fontsize=15,y=.98)
fig.text(.5,.025,'60 块 TRAIN 留出地形 · 117 个预选窗口 · 只统计全部有效供体直达功率为正的位置\n红点排除了同一组供体上的任意凸插值权重，尚未排除换网格、TX 或供体映射。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.115,1,.94))
fig.savefig(O/'C94供体上界与标签的矛盾_20261103.png')
plt.close(fig)
print('report and figure generated',counts)
