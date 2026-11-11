from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];audit=root/'audit'
r=json.loads((audit/'raw_audit.json').read_text());v=json.loads((audit/'verification.json').read_text());sp=json.loads((audit/'internal_split.json').read_text())
assert v['all_passed']
lines=['# 原始数据独立检查','', '日期：2026-10-31。检查在 transformer 上执行。没有训练模型，没有使用 GPU，没有提交预测。','', '## 结论','', '顶层发布格式数据的结构、索引、全量数值和掩码一致性检查通过。检查覆盖训练集与官方验证集。没有发现阻止数据读取和独立基线准备的结构或数值异常。尚未与官方校验和比较，不能将这一结果写成“与官方发布文件逐字节一致”。','', '原始数据只读。没有读取或使用另一 AI 的代码、方案、模型、特征或实验结果。没有读取恢复版和试验版数据。','', '## 实际检查范围','', f"- 高度图：{sum(x['terrains'] for x in r['splits'].values()):,} 张，每张 256×256。",f"- 路损图：{2*sum(x['samples'] for x in r['splits'].values()):,} 张，包含两个频段。",'- 掩码：所有训练与官方验证样本、两个频段全部解包检查。','- 索引：全部样本的 ID、发射机坐标、地形归属和高度图行映射。','- 当前提交模板：800,000 行 ID 全部核对，不生成预测、不上传。','- 原始文件：为本轮实际读取的 18 个发布格式文件保存 SHA256；没有读取官方加载器代码。','', '## 数据量与有效像素','', '表中的路损图数量为每个频段的样本数。有效像素比例的分母是该划分、该频段的全部像素。','', '| 划分 | 地形数 | 每频段样本数 | 415 MHz 有效像素 | 5.8 GHz 有效像素 |','|---|---:|---:|---:|---:|']
for s,label in [('train','训练集'),('val','官方验证集')]:
 a=r['splits'][s]
 lines.append(f"| {label} | {a['terrains']:,} | {a['samples']:,} | {100*a['bands']['415']['valid_fraction']:.4f}% | {100*a['bands']['58']['valid_fraction']:.4f}% |")
lines+=['', '当前顶层发布文件中未发现测试输入。测试样本数量仅来自官方计划，未作为本地已有数据计入。','', '## 完整性与对应关系','', '- 训练集与官方验证集的样本 ID、地形 ID 均无交集。','- 所有高度图做内容哈希。划分内没有完全重复的高度图，两个划分之间也没有完全相同的高度图。','- 官方 val_terrains.txt 中的 680 个地形与验证索引完全一致。','- 高度图没有 NaN、Inf 或常数图。每张高度图的最小值均为 0 米。','- 四个路损数组都没有 NaN、Inf 或非正值。没有空有效掩码，也没有有效区域为常数的路损图。','- 所有有效路损像素都落在 metadata.json 声明的对应频段范围内。','- 5.8 GHz 的全部 mask=0 像素都恰好为 145 dB。','- 提交模板全部 ID 都能映射到官方验证集。没有重复 ID。每频段 100 张图，每张图 4,000 个像素，所有选中像素均为 mask=1。','', '另对 256 个训练样本和 64 个验证样本检查坐标。两个频段的最低有效路损像素都在发射机 3 米内。距离中位数均为 1 米。相反，行翻转、列翻转或转置坐标后，距离中位数明显增大。这是抽样的坐标一致性证据，不是对全部像素的物理正确性证明。','', '## 对参赛方案有影响的事实','', '### 1. 145 dB 不能作为无效像素判断条件','']
valid145=sum(r['splits'][s]['bands']['58']['valid_eq145_pixels'] for s in ['train','val'])
inv=sum(r['splits'][s]['bands']['58']['total_pixels']-r['splits'][s]['bands']['58']['valid_pixels'] for s in ['train','val'])
tot=sum(r['splits'][s]['bands']['58']['total_pixels'] for s in ['train','val'])
lines +=[f'训练集有 {r["splits"]["train"]["bands"]["58"]["valid_eq145_pixels"]:,} 个有效像素等于 145 dB。官方验证集有 {r["splits"]["val"]["bands"]["58"]["valid_eq145_pixels"]:,} 个。两者合计 {valid145:,} 个。直接按数值过滤会错误删除这些真实标签。','',f'两划分合计的 5.8 GHz 无效像素比例为 {100*inv/tot:.4f}%。这是本轮对全部像素的计数。网页中的约 11.4% 是另一条概略描述，不能替代本次数值，也不能仅凭差别断定文件损坏。', '',f'最少有效像素的 5.8 GHz 训练图只有 {100*r["splits"]["train"]["bands"]["58"]["valid_fraction_distribution"]["0"]:.4f}% 像素参与评分。因此损失必须按有效像素计数归一化。','', '### 2. 样本数与地形数严重不均衡','', '| 划分与分组 | 地形数 | 每频段样本数 | 样本占该划分比例 |','|---|---:|---:|---:|']
for s,label in [('train','训练'),('val','官方验证')]:
 a=r['splits'][s]
 for g,gl in [('multi','多发射机'),('single','单发射机')]:
  lines.append(f"| {label}／{gl} | {a['terrain_groups'][g]:,} | {a['sample_groups'][g]:,} | {100*a['sample_groups'][g]/a['samples']:.2f}% |")
lines+=['', '因此开发集必须按地形划分。总体 RMSE 之外，要报告两类地形的结果。暂不把地形均衡重采样设为默认训练规则，因为它会改变训练目标的权重。','', '### 3. 两个频段相关，但差值并非常数','']
for s,label in [('train','训练集'),('val','官方验证集')]:
 a=r['splits'][s]['cross_band_common_valid']
 lines.append(f"- {label}：在 {a['pixels']:,} 个双频共同有效像素上，相关系数为 {a['pearson_all_pixels']:.4f}。5.8 GHz 减去 415 MHz 的均值为 {a['delta_58_minus_415_mean_db']:.3f} dB，标准差为 {a['delta_58_minus_415_std_db']:.3f} dB。")
lines+=['', '这些是描述统计。距离和共同地形本身就可能带来相关性。它们支持测试共享表示，但不能证明共享模型更好，也不支持用固定频率偏移代替高频预测。','', '### 4. 地形和路损尾部不能随意裁剪','', '| 划分 | 地形高差中位数／最大值（m） | 415 MHz 有效值范围（dB） | 5.8 GHz 有效值范围（dB） |','|---|---|---|---|']
for s,label in [('train','训练集'),('val','官方验证集')]:
 a=r['splits'][s];h=a['heightmaps']['relief_distribution'];lo=a['bands']['415']['valid_pixel_distribution']['quantiles'];hi=a['bands']['58']['valid_pixel_distribution']['quantiles']
 lines.append(f"| {label} | {h['0.5']:.3f}／{h['1']:.3f} | {lo['0']:.5f}–{lo['1']:.3f} | {hi['0']:.5f}–{hi['1']:.3f} |")
lines+=['', '第一版不添加按经验阈值硬裁剪目标的处理。归一化参数仅从训练资料确定，并保留线性输出。','', '## 已生成的开发划分','', '只从官方训练索引中按地形分层。开发验证集包括 39 块多发射机地形和 573 块单发射机地形。没有官方验证样本进入开发训练集。','', f"- 开发训练：{sp['summary']['fit']['terrains']:,} 块地形，{sp['summary']['fit']['samples_per_band']:,} 个样本/频段。",f"- 开发验证：{sp['summary']['dev']['terrains']:,} 块地形，{sp['summary']['dev']['samples_per_band']:,} 个样本/频段。",'- 随机种子：20261031。完整地形名单及来源索引哈希：`audit/internal_split.json`。','', '## 核验与限制','', f"`audit/verification.json` 的 {len(v['checks'])} 项交叉核验全部通过。有效像素均值分别通过像素直方图和逐图加权统计计算，两种方法一致。掩码解包另外通过位运算及重新打包检查。",'', '- 本地 SHA256 未与官方可信校验和对照。','- 无法从匿名地形 ID 核验更大父地形之间的关联。没有完全重复不等于统计独立。','- 坐标检查是 320 个样本的抽查。全量数值扫描不等于逐像素物理真值核验。','- 本轮没有模型训练、模型推理或性能比较。不能从数据检查推出任何模型得分。','', '检查期间曾因 val_terrains.txt 的说明注释导致汇总脚本中断。已修正解析器并重新完成全量检查。最终统计来自修正后的成功运行。','', '## 文件','', '- `audit/raw_audit.json`：完整汇总和 18 个源文件的 SHA256。','- `audit/map_stats.csv`：56,100 张路损图的统计。','- `audit/train_terrain_stats.csv`、`audit/val_terrain_stats.csv`：逐地形统计及内容哈希。','- `audit/input_alignment.json`：抽样名单、坐标诊断和掩码位检查。','- `audit/verification.json`：交叉核验结果。','- `audit/audit.log`：最终成功运行的日志。','- `参赛方案.md`：方法、对照、评价和提交计划。','']
(root/'数据检查.md').write_text('\n'.join(lines))
print('wrote',root/'数据检查.md','checks',len(v['checks']),'invalid58_pct',100*inv/tot,'valid145',valid145)
