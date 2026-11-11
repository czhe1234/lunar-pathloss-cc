"""Consolidate measured C5 result, negative decision, and provenance. CPU only."""
from pathlib import Path
import json,hashlib,datetime,csv
R=Path(__file__).resolve().parents[1];D=R.parent/'data';run=R/'runs/c5_cell_observation_s20261101';P=run/'predictions/train_dev_bf16';B=R/'runs/c4_critical_length_fast_s20261031/predictions/train_dev_bf16';O=R/'audit/c5_final';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
a=read(P/'independent_score.json');b=read(B/'independent_score.json');assert a['all_verification_checks_passed']
assert read(P/'indices.json')==read(B/'indices.json')
comparison={}
for band in ['415','58']:
 x=b['bands'][band]['rmse_db'];y=a['bands'][band]['rmse_db'];assert a['bands'][band]['valid_pixels']==b['bands'][band]['valid_pixels']
 comparison[band]={'C4_dev_rmse_db':x,'C5_dev_rmse_db':y,'absolute_reduction_db':x-y,'relative_reduction_percent':100*(x-y)/x,'valid_pixels':a['bands'][band]['valid_pixels']}
config=read(run/'config.json');checks=[]
for n,v in config['source_sha256'].items():checks.append({'name':'frozen source '+n,'passed':sha(run/'source'/n)==v})
raw=read(R/'audit/raw_audit.json')
for n,v in raw['files'].items():
 s=(D/n).stat();checks.append({'name':'raw data size/mtime '+n,'passed':s.st_size==v['size_bytes'] and s.st_mtime_ns==v['mtime_ns']})
old_sha='4355f5e831aa280bf502c01488f823d67eee389e769e8d616644fa9d9a9ca907'
checks.append({'name':'C4 full checkpoint unchanged','passed':sha(R/'runs/c4_full_refit_s20261031/best.pt')==old_sha})
old_dev_sha=read(B/'inference.json')['checkpoint_sha256'];checks.append({'name':'C4 development checkpoint unchanged','passed':sha(R/'runs/c4_critical_length_fast_s20261031/best.pt')==old_dev_sha})
new_sha=sha(run/'best.pt');checks.append({'name':'C5 prediction checkpoint matches current best','passed':new_sha==read(P/'inference.json')['checkpoint_sha256']})
checks.append({'name':'C5 initialized with C4 development checkpoint','passed':read(run/'initialization.json')['checkpoint_sha256']==old_dev_sha})
checks.append({'name':'input cache and target transport checks passed','passed':read(run/'input_cache_replay_check.json')['passed'] and read(run/'target_transport_check.json')['passed']})
checks.append({'name':'CPU/GPU independent scoring and replay passed','passed':a['all_verification_checks_passed']})
checks.append({'name':'single permitted GPU recorded in train and predict','passed':config['device']['uuid']==read(P/'inference.json')['device']['uuid']=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'})
assert all(z['passed'] for z in checks)
result={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision':'C5 rejected as a substantial-improvement version; keep C4. No full refit or C5 official-validation evaluation.','development_maps_per_band':2562,'development_terrains':len({r['terrain_id'] for r in csv.DictReader((D/'train_index.csv').open()) if r['terrain_id'] in set(read(R/'audit/internal_split.json')['dev_terrain_ids'])}),'comparison':comparison,'best_epoch':2,'completed_epochs':6,'checkpoint_sha256':new_sha,'checks':checks,'all_checks_passed':True,'large_gain_target_met':False,'official_1dB_goal_met':False,'earlier_official_val_labels_used_in_diagnosis':True}
(O/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (O/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['band']+list(comparison['415']));w.writeheader();w.writerows([{'band':k,**v} for k,v in comparison.items()])
hist=read(run/'history.json');ra=read(P/'region_score.json');rb=read(B/'region_score.json');tri=read(R/'audit/c5_exact_mesh_triage/summary.json');inf=read(P/'inference.json')
lines=['# C5 新版本试验：未通过大幅提分门槛','',
'日期：2026-11-01，Asia/Shanghai。用户本轮重新授权实施提分版本。所有新增文件在服务器 CC，训练和推理只用实体 3 的完整 UUID。没有读取或复用另一个 AI 的实现、模型或特征。','',
'## 结论','',
'**本轮没有得到大幅提分版本。C5 已实现并完成 6 轮开发训练；独立复算后双频只改善约 0.15% / 0.13%。按开训前门槛停止，不推进全量再拟合，不替换 C4。**','',
'此前确认的是官方观测流程，尚未找到一个已证实、修复后能大幅降低误差的生成错误。本轮把单元可见比例、功率平均和插值显式放入模型，仍没有得到实质收益。这个负结果不能证明所有观测模型无效，也不能据此宣布双频 1 dB 不可达。','',
'## 同一开发集上的独立复算','',
'两版均使用原固定开发划分：2,562 张双频图、612 块地形。C5 从在 FIT 上训练、由开发集选出的 C4 检查点初始化，所有梯度只使用 FIT。这里的分数不是官方固定验证分数。','',
'| 频段 | C4 RMSE dB | C5 RMSE dB | 降低 dB | 相对降低 | 有效像素数 |','|---|---:|---:|---:|---:|---:|']
for band,label in [('415','415 MHz'),('58','5.8 GHz')]:
 z=comparison[band];lines.append(f"| {label} | {z['C4_dev_rmse_db']:.6f} | {z['C5_dev_rmse_db']:.6f} | {z['absolute_reduction_db']:.6f} | {z['relative_reduction_percent']:.3f}% | {z['valid_pixels']:,} |")
lines+=['',
'开训前规定：双频各改善 20% 才称大幅；第 6 轮若双频未各改善 5%，停止延长训练。C5 两项门槛均未通过。最好检查点是第 2 轮。','',
'当前保留的 C4 完整官方固定验证 RMSE 仍是 3.812677 / 4.990175 dB，目标未达。本轮没有 C5 的完整官方验证分数，没有网站提交。','',
'## 改了什么','',
'保留 C4 输入与编码器，在输出端增加接收单元观测模块：对像素周围 3×3 子像素做可见性近似；学习有界的阈值修正、位置偏移和过渡宽度；分别预测可见部分与遮挡部分的条件功率；在功率平均与 dB 插值之间学习组合。两频段共享可见比例，使用各自的条件功率。','',
'这没有恢复实际三角网格或实际 TX 坐标。可见性修正不是实际 TX 高度估计，两个条件功率也不是逐路径真值。掩码只用于损失和评分，没有作为输入。具体方程与预定参数见 [实施协议](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/results/C5接收单元观测模型_协议_20261101.md)。','',
'## 完整 6 轮结果','',
'下表来自训练评价器。最终结论使用上表独立输入推理后的 NumPy 复算。前 2 轮只更新观测模块，后 4 轮联合更新。','',
'| 轮次 | 更新范围 | 开发 415 MHz RMSE | 开发 5.8 GHz RMSE |','|---:|---|---:|---:|']
for z in hist:lines.append(f"| {z['epoch']} | {'观测模块' if z['epoch']<=2 else '联合微调'} | {z['dev']['rmse415']:.6f} | {z['dev']['rmse58']:.6f} |")
lines+=['','联合微调没有超过第 2 轮。没有把第 6 轮较低的训练内误差当作验证改善。','',
'## 主要误差区是否改善','',
'三个分区由输入几何假设定义，不是真实传播机制标注。E3 是 TX 3 m 假设下的最大遮挡裕量；E2 是 TX 2 m 敏感性假设下的裕量。','',
'| 几何分区 | 415 MHz：C4 → C5 | 5.8 GHz：C4 → C5 |','|---|---:|---:|']
for key,label in [('both_clear_geometry','两种高度假设均可见'),('height_sensitive_geometry','高度敏感区：E3<0 且 E2≥0'),('blocked_at_3m_geometry','3 m 假设下遮挡：E3≥0')]:
 lines.append(f"| {label} | {rb['bands']['415'][key]['rmse']:.4f} → {ra['bands']['415'][key]['rmse']:.4f} | {rb['bands']['58'][key]['rmse']:.4f} → {ra['bands']['58'][key]['rmse']:.4f} |")
lines+=['',
'敏感区双频只改善约 0.02 dB；高频遮挡区略有退步。当前结果不支持“单元观测模块已修正主要误差”。','',
'## 避免继续堆同类训练的补充核对','',
'在 64 块不同 FIT 地形、每图固定 1,024 像素上，直接按两种假设的规则三角划分计算精确线段最大遮挡。每条线段检查网格边和三角对角线的交点，并包含端点；平面解析检查通过。没有训练或传播功率仿真。','',
'两种三角划分与原 256 点双线性剖面的可见/遮挡符号分别只有 26/65,536（0.0397%）和 21/65,536（0.0320%）不同。现有证据不足以为“只把剖面换成更精细三角剖面”再开一轮完整训练。此检查是假设网格上的比较，不能验证组织者实际网格，也不能把小范围结果当作全数据误差上界。','',
'## 效率、数值与来源核验','',
f"- 实体 3 H100，完整 UUID 在配置与推理记录一致。批次 32；预热后的编译短测为 406.4 图/秒，实际稳定训练轮约 71 秒/轮。训练峰值显存约 {hist[-1]['peak_memory_GB']:.2f} GB。48/64 批次因显存不足未采用。初始输入加载曾受磁盘读取限制，训练期间复用显存缓存。",
f"- 独立推理只读取高度图和 TX 索引。单样本预热后平均 {inf['single_request_warm_ms']['mean']:.2f} ms、p95 {inf['single_request_warm_ms']['p95']:.2f} ms；首次调用 {inf['single_request_cold_ms']:.1f} ms，均不含模型载入。这是本机测试，不是官方硬件验收。",
'- 同一份预测的 NumPy/Torch float64 计分最大差为 2.7e-15 dB。独立推理与训练评价器最大差为 0.0000195 dB，通过既定复算门槛。',
f"- {len(checks)} 项来源与结果检查通过。原始数据大小/mtime、C4 开发权重和 C4 全训练权重未变。输入缓存重放、目标和掩码传输检查通过。",
f'- C5 最好权重 SHA256：`{new_sha}`。',
'- 原始失败记录保留：第一次资源试跑的 buffer 内存格式错误已修正；更大批次的 OOM 仅发生于丢弃权重的资源试跑。正式训练没有非有限损失或 OOM。',
'- 训练随机种子与原对照一致。本轮是单种子、单候选试验；没有给出多种子收益声明。官方验证标签此前已用于事后诊断，本轮没有用它更新权重或挑 epoch。','',
'## 交付入口与停止状态','',
'- [结果汇总与核验](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/audit/c5_final/summary.json)',
'- [结果表 CSV](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/audit/c5_final/comparison.csv)',
'- [C5 模型](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/src/c5_model.py)',
'- [C5 训练入口](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/scripts/c5_train.py)',
'- [只读输入推理入口](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/scripts/c5_predict.py)',
'- [候选权重，未晋升](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/runs/c5_cell_observation_s20261101/best.pt)',
'- [独立开发计分](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/runs/c5_cell_observation_s20261101/predictions/train_dev_bf16/independent_score.json)',
'- [精细几何核对](/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/audit/c5_exact_mesh_triage/summary.json)','',
'本轮结束后停止。C5 代码与权重用于复核这个候选，不作为已达到大幅提分要求的新主版本。没有全量再拟合或后续训练队列。']
report=R/'results/C5新版本试验与停止结论_20261101.md';report.write_text('\n'.join(lines)+'\n')
print(json.dumps({'comparison':comparison,'checks_passed':len(checks),'report':str(report)},ensure_ascii=False))
