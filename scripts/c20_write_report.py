from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parents[1];O=R/'audit/c20_cell_observation_20261101'
r=json.loads((O/'results.json').read_text());v=json.loads((O/'verification.json').read_text());n=json.loads((O/'extended_numerics.json').read_text());meta=json.loads((O/'observations.json').read_text())
def val(b,name,z='occluded_proxy'):return r['statistics'][b][name][z]
def ci(a):return f"[{a[0]:+.3f}, {a[1]:+.3f}]"
lines=['# C20：接收单元面积平均与插值对照','',
'日期：2026-11-01。全部新增代码、计算、统计及报告在服务器 CC。未训练新的参赛模型，未使用 GPU。只拟合了少量诊断校准系数。没有读取另一 AI 的实现或结果。','',
'**结论：面积平均确实改变候选物理传播图，但本轮不能把它转化为对现有 C10 的可靠大幅改进。主候选在内部确认集遮挡区的平方误差只名义下降 0.368% / 0.199%，两频置信区间均包含零。扣除已有局部起伏控制后的额外下降为 0.200% / 0.251%，区间也包含零。因此不启动新训练。**','',
'这个结论针对本轮明确构造的两个 1 m 候选网格和传播近似。它不是原始仿真网格的因果归因结果，不能据此宣布实际面积平均只占总误差的 0.37%，也不能据此证明输入存在不可约信息缺口。','',
'**固定数据与比较方法。** 沿用 C19 的 612 个地形、每地形一个按 ID 选定的 TX，分为 306 个校准地形与 306 个确认地形。为控制 CPU 开销，在看本轮结果前，固定取原来均匀抽样 256 个查询点的前 128 个。每个候选使用同一批点。两部分地形互不重叠，均不参与 C10 权重拟合；但 DEV 曾用于历史模型选择，不能称为新的盲测。官方验证不参与本轮。','',
'确认集遮挡区有效查询点数为低频 11,276、高频 7,249。有效掩码只用于评价，不进入点功率、单元平均或插值计算。本文所有百分比都是 SSE（平方误差和）变化，不能直接解释为 RMSE 百分比。','',
'遮挡区仍按 C19 的输入几何代理定义：名义 TX 2 m 时最大净空不小于零。高度敏感区为 TX 3 m 清楚、2 m 遮挡；名义遮挡区为 TX 3 m 也遮挡。“五点均遮挡”还要求查询点及四个对角邻点均为名义遮挡。另记录候选接收单元中部分有 LOS 的区域，但它不是实际射线标签。','',
'**候选网格和真正改变的变量。** 以发布的 256×256 高度值作为整数网格顶点，按两种相反的对角线剖分；接收网格整体向 z 方向平移 1 m。TX 采用已有候选坐标：整数行列，z=floor(Htx)+3 m。这个 TX 和两个网格都不是已找回的原始仿真记录。','',
'对每个固定候选网格，保持传播模型不变，分别输出：查询位置点功率；三角中心点功率经插值；单元平均功率经插值。直接透射部分采用解析三角面求交和有序 V 极化薄层传输；散射部分沿用输入几何的一次漫反射积分近似。它不包括所有多次反射/散射路径，不能视为实际点功率真值。','',
'接收单元平均严格在线性功率域计算：Gcell=(1/A)∫triangle G(x)dA。每个平面三角形的面积雅可比是常数，因此等面积重心子三角的采样权重相等；没有把 1/(A·|n·k|) 再乘成传播增益。官方该因子属于射线积分估计的归一化。[Sionna RT 1.1.0 固定源码](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/mesh_radio_map.py#L117-L145)。','',
'主候选：剖分 0、每单元 16 个等面积采样点、对单元平均线性功率做重心插值，最后转为 dB。对照包括每单元 4/64 点、另一剖分、只插值三角中心点，以及先将单元均值转 dB 再插值。后者是导出顺序敏感性检查，不能认定它就是官方实际流程。所有三角中心均保留；原流程若有缺失单元后再做插值，所需原始单元有效性尚未获得。','',
'**一个可以精确验证的结果。** 对这两种规则候选网格，若三角面内的功率按顶点值线性变化，则“单元面积平均→中心线性插值”在内部整数网格点都化为：','',
'Gout = G中心/3 + (G上 + G下 + G左 + G右)/6。','',
'这表明该条件下观测算子是一个局部平滑。常数、仿射场及任意顶点分片线性场均通过解析与数值一致性检查。真实遮挡造成的面内非线性由 16/64 点积分另行检验，不能用这个公式代替全部传播计算。','',
'**如何测量对现有误差的解释力。** C10 已学习平均、插值后的发布标签，因此没有把它冒充真实点功率。先从同一候选传播场计算 ΔL=L单元平均插值−L点，再用 P新=P_C10+β·ΔL 检验剩余误差。每频每输入分区只用校准地形确定一个 β，限制在 [0,1]；ΔL 固定截断到 ±30 dB，避免不完整散射近似中的极端值支配校准。原始未截断增量分布完整保留。确认集不选 β、网格或超参数。','',
'另在既有距离、净空、附加路损和 Q(P) 的小型控制之后检查 ΔL 的额外收益。主候选预先固定，表中不选择最好的剖分。多对照区间未做多重比较校正；不以单个略高于零的对照作为推进依据。','',
'| 对照 | 415 MHz 遮挡 RMSE dB | 高频遮挡 RMSE dB | 415 SSE 下降及 95% 区间 | 高频 SSE 下降及区间 |',
'|---|---:|---:|---:|---:|']
labels={'C10':'冻结 C10','triangle0_centroid':'只做单元中心插值','triangle0_area_increment':'仅面积平均增量','triangle0_area16':'主候选：面积平均 + 插值','triangle1_area16':'另一三角剖分','triangle0_area16_dbinterp':'单元均值转 dB 后插值','triangle0_area16_fixed_diffuse':'固定散射背景对照','curvature_control':'已有局部起伏控制','triangle0_area16_after_control':'已有控制 + 主候选','C10_fivepoint_dB':'C10 五点 dB 平滑','C10_fivepoint_power':'C10 五点线性功率平滑'}
for name,label in labels.items():
 a,b=val('415',name),val('58',name)
 lines.append(f"| {label} | {a['confirm_RMSE']:.5f} | {b['confirm_RMSE']:.5f} | {a['SSE_gain_percent']:+.3f}% {ci(a['SSE_gain95'])} | {b['SSE_gain_percent']:+.3f}% {ci(b['SSE_gain95'])} |")
lines+=['',
'面积平均加插值，相比只做中心插值，额外下降 0.229% / 0.141% SSE，区间仍跨零。相对已有局部起伏控制，主候选的额外下降为 0.200% / 0.251%，95% 区间分别为 [−0.080,0.476] / [−0.045,0.518]%。因此不能说发现了稳定、独立的物理修正收益。','',
'五点 dB 平滑的 2.105% / 2.777% SSE 下降属于对既有预测的轻微后处理收益。它不是线性功率面积平均。把 C10 再按线性功率平均，高频反而增加 4.438% SSE；这说明不应把“重建观测过程”简化为给当前输出再加一遍功率平均。','',
'**边界与遮挡内部。** 下表仅列主候选，相同子集内比较；“部分 LOS 单元”由候选网格积分点确定，不使用真值划区。','',
'| 输入子区 | 415 点数 / C10→修正 RMSE | 高频点数 / C10→修正 RMSE | 415 / 高频 SSE 下降 |',
'|---|---:|---:|---:|']
for z,label in [('height_sensitive','高度敏感区'),('strict_nominal_blocked','中心及四邻点均名义遮挡'),('candidate_partial_LOS','候选单元部分有 LOS'),('candidate_no_LOS','候选单元均无 LOS 且在遮挡区')]:
 a,b=val('415','triangle0_area16',z),val('58','triangle0_area16',z);aa,bb=val('415','C10',z),val('58','C10',z)
 lines.append(f"| {label} | {a['confirm_n']:,} / {aa['confirm_RMSE']:.3f}→{a['confirm_RMSE']:.3f} | {b['confirm_n']:,} / {bb['confirm_RMSE']:.3f}→{b['confirm_RMSE']:.3f} | {a['SSE_gain_percent']:+.3f}% / {b['SSE_gain_percent']:+.3f}% |")
lines+=['',
'主候选对五点均遮挡区域只名义下降 0.680% / 0.067% SSE。没有出现“边界一平均就大幅修复”或“遮挡内部由观测平均统一解释”的结果。对应各分区置信区间均保存在 results.json，不以这个子区表作新候选选择。','',
'**面积平均对粗物理模型本身仍有影响。** 同一确认遮挡子集，低频候选物理场的原始 RMSE 为：点值 9.248 dB、中心插值 7.927 dB、单元平均加插值 7.234 dB。这个变化真实存在，但 C10 原来已是 4.506 dB。因此不能把粗物理场的改善直接当成对 C10 剩余误差的改善。','',
'高频对应值为 30.253→28.854→26.761 dB，远差于 C10 的 6.193 dB，且存在下面的数值不稳定。这说明本轮一次散射场不足以充当高频点功率真值；它的平均改善不能用来证明实际标签生成误差占比。','',
'**数值精度与解释边界。** 在事先按 ID 选出的 64 个校准地形、每图 16 个均匀查询点上，另算每单元 64 点与 64 个角方向。该选择不看标签；有效掩码仅用于汇报计分范围内的数值误差。主候选遮挡有效点为低频 324、高频 174。','',
'| 数值对照，主剖分 | 415 RMS / 绝对误差 p95 dB | 高频 RMS / p95 dB |',
'|---|---:|---:|']
for k,label in [('area16_to_area64','单元 16→64 点：平均功率结果变化'),('angular32_to64_delta','角方向 32→64：观测增量变化')]:
 a=n['checks']['triangle0_full_'+k+'_415_occluded'];b=n['checks']['triangle0_full_'+k+'_58_occluded']
 lines.append(f"| {label} | {a['RMS_dB']:.3f} / {a['abs_p95_dB']:.3f} | {b['RMS_dB']:.3f} / {b['abs_p95_dB']:.3f} |")
lines+=['',
'低频多数位置较稳定，但仍有少数积分难点；高频一次散射角积分未收敛，32→64 方向时约 10.9% 的遮挡有效检查点，其观测增量相差超过 1 dB，最大约 89 dB。没有因为不稳定而删除这些确认点，也没有把这类变化解释成真实传播效应。','',
'固定散射背景对照把每个查询点的散射功率在邻近单元内保持不变，只让直达/透射分量参与空间平均。其遮挡 SSE 名义收益也只有 0.366% / 0.255%。另外允许逐像素用真值选择最有利的 β 和两种剖分，只作为诊断上限：在该固定背景、±30 dB 有限修正族内，高频五点均遮挡区域的最大 SSE 下降也仅 0.205%。这个上限不能推广到真实空间变化的散射场，更不是数据的不可约误差下界。','',
'**判断与后续依据。** 本轮不支持把“再平均一次、换一种插值、换一条三角对角线”作为主要提分方向。保留面积平均和插值作为未来传播计算必须匹配的观测定义，但不据当前结果训练新模型。单纯 dB 平滑的收益仍很小，远不足以达到双频 1 dB。','',
'若继续推进，首先需要改进并检验候选点功率场，尤其是高频的到达路径和数值积分；在点功率场本身远差于现有模型、且角积分不稳定时，再拟合复杂观测修正无法回答主要物理误差来自哪里。原始网格的尺寸、坐标相位、实际 TX、缺失单元处理和插值域仍未恢复。本轮没有检验所有可能的非规则网格，也没有证明面积平均在真实生成链中不重要。','',
f"复核通过：{v['saved_correction_RMSE_checks']} 项保存修正量的独立 RMSE 重算，最大差 {v['max_RMSE_difference']:.3e} dB；从原始功率重建保存 dB 值最大差 {v['raw_power_to_saved_dB_max_error']:.3e} dB；两张图的独立串行物理计算与批量保存功率逐项一致。18 个原始文件大小/mtime 和四份 C10 权重 SHA256 不变。主提取约 {meta['seconds']:.1f} 秒，数值扩展约 {n['seconds']:.1f} 秒，使用 8 个 CPU 工作线程。",'',
'初次批量运行在共享插值对象并发初始化阶段异常退出（139），未使用其部分数组。改为串行预计算全部重心插值索引和权重后，批量完成；随后串行重算一致。初次失败文件、退出码和处理记录保留，不将失败产物计入结果。','',
'当前官方固定验证版本仍为 C10：2.967202 / 3.387979 dB；官方综合分仍为 3.1775908266 dB。本轮没有新官方评分或网站提交。','',
'证据目录：/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/audit/c20_cell_observation_20261101/。','',
'- protocol.json、evaluation_plan.json：假设、主候选、对照和推进门槛。',
'- selection.json、pixels.npy：固定地形/TX/像素。',
'- operator_loss.npy、component_powers.npy、observations.json：各观测算子结果和来源哈希。',
'- results.json、calibrated_corrections.npz、calibration_coefficients.json：全部分区、正负对照、校准系数和可复算修正。',
'- implementation_checks.json、extended_numerics.json、verification.json：解析、数值收敛与独立复核。',
'- failed_parallel_initialization/、source/：保留的失败记录及完成版源码。','',
'在 CC 复算：','',
'~~~sh',
'TMPDIR="$PWD/.cache/tmp" g++ -O3 -std=c++17 -fPIC -shared src/c20_triangle_transport.cpp -o src/c20_triangle_transport.so',
'.venv/bin/python scripts/c20_validate_and_pilot.py',
'.venv/bin/python scripts/c20_build_observations.py',
'.venv/bin/python scripts/c20_evaluate.py',
'.venv/bin/python scripts/c20_extended_numerics.py',
'.venv/bin/python scripts/c20_verify.py',
'~~~','']
dest=R/'results/C20接收单元面积平均与插值对照_20261101.md';dest.write_text('\n'.join(lines))
p=R/'STATUS.md';s=p.read_text();a=s.index('当前工作：C20');b=s.index('已核验版本：C10',a)
s=s[:a]+'''当前工作：C20接收单元面积平均与插值对照已完成。612个内部DEV地形、306/306校准/确认，每图128个固定均匀查询点；两种1m三角网格；没有训练新参赛模型、没有使用GPU。
主候选面积平均+插值对C10遮挡SSE名义下降0.368%/0.199%，95%区间均跨零；扣除已有局部起伏控制后的额外下降0.200%/0.251%，仍跨零。
面积平均改善了粗物理场本身，但没有转化成C10大幅收益。高频一次散射点功率场存在角积分不稳定，不能据此定量归因真实标签生成误差。
没有候选通过5%遮挡SSE及数值可靠性门槛，不启动新训练。保留观测算子定义，下一项依据应先来自可靠的点功率/到达路径计算。
报告：results/C20接收单元面积平均与插值对照_20261101.md；证据：audit/c20_cell_observation_20261101/。
560项独立RMSE复算通过；串行重算与批量功率一致；原始文件状态及四份C10权重不变。官方版本及综合分3.1775908266dB未变。
C19证据保留在results/C19细尺度传播与地形标签对应检验_20261101.md。

'''+s[b:];p.write_text(s)
shutil.copy2(Path(__file__),O/'source'/Path(__file__).name)
manifest={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (O/'source').iterdir() if f.is_file()}
manifest[str(dest.relative_to(R))]=hashlib.sha256(dest.read_bytes()).hexdigest()
(O/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'report':str(dest),'bytes':dest.stat().st_size,'verification':v['state'],'status_updated':True},ensure_ascii=False))
