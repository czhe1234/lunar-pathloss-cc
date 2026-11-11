"""Create the C10 release only after complete, independently scored validation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,csv,shutil,tarfile
R=Path(__file__).resolve().parents[1];run=R/'runs/c10_full_refit_s20261101';pred=run/'predictions/val_full_gpu_c10_fast_fresh'
dev=R/'runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
off=read(pred/'independent_score.json');ds=read(dev/'independent_score.json');inf=read(pred/'inference.json');fit=read(run/'result.json');cfg=read(run/'config.json')
assert off['complete_requested_split'] and off['maps_per_band']==2330 and ds['complete_requested_split'] and ds['maps_per_band']==2562
assert inf['no_target_or_validity_mask_read'] and inf['fresh_geometry'] and inf['GPU_used']['uuid']=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
checks={'complete_official_validation':True,'complete_internal_development':True,'input_only_inference':True,'only_approved_GPU':True}
for b,n in [('415',151537682),('58',135926253)]:
 assert sha(run/f'band{b}.txt')==fit['bands'][b]['sha256']==inf['model_sha256'][b]
 assert off['bands'][b]['valid_pixels']==n and off['bands'][b]['same_prediction_scorer_difference_db']<1e-8
 checks[b+'_weight_hash_and_scorers']=True
for name,ref in read(R/'audit/raw_audit.json')['files'].items():
 st=(R.parent/'data'/name).stat();assert st.st_size==ref['size_bytes'] and st.st_mtime_ns==ref['mtime_ns']
assert sha(R/'runs/c4_full_refit_s20261031/best.pt')=='4355f5e831aa280bf502c01488f823d67eee389e769e8d616644fa9d9a9ca907'
assert read(R/'audit/c10_standalone_verification.json')['passed']
checks.update(original_raw_file_stats_unchanged=True,C4_weights_unchanged=True,standalone_interface_verified=True)
zones=read(R/'audit/c10_official_error_zones.json')
score=lambda b:off['bands'][b]['rmse_db'];base=lambda b:off['C4_reference'][b]['rmse_db'];gain=lambda b:off['C4_reference'][b]['relative_RMSE_reduction_percent']
target='已达到' if off['goal_met_on_official_validation'] else '尚未达到'
pkg=R/'releases/C10_20261101';pkg.mkdir(parents=True,exist_ok=False)
for n in ['models','src','scripts']:(pkg/n).mkdir()
(pkg/'src/__init__.py').write_text('')
for b in ['415','58']:shutil.copy2(run/f'band{b}.txt',pkg/'models'/f'band{b}.txt')
for n in ['cc_core.py','c3_geometry.py','c4_geometry.py','gpu_forest.py','c7_surface_features.py','c8_tx_context.py','c9_extra_gpu.py','c10_phase_geometry.py','c10_inference.py']:
 shutil.copy2(R/'src'/n,pkg/'src'/n)
shutil.copy2(R/'scripts/c10_predict_one.py',pkg/'scripts/c10_predict_one.py')
for p,n in [(run/'config.json','training_config.json'),(pred/'independent_score.json','official_validation_score.json'),(dev/'independent_score.json','development_validation_score.json'),(pred/'inference.json','inference_measurement.json'),(R/'requirements.c6.lock.txt','requirements.lock.txt')]:shutil.copy2(p,pkg/n)
row=next(csv.DictReader((R.parent/'data/val_index.csv').open()))
cmd=f"./run_gpu3.sh releases/C10_20261101/scripts/c10_predict_one.py --model-directory models --heightmap {R.parent}/data/val_hm.npy --hm-row {row['hm_row']} --tx-row {row['tx_row']} --tx-col {row['tx_col']} --output example_prediction.npy"
readme=f"""# C10 服务器复现包

完整官方固定验证：415 MHz {score('415'):.6f} dB；5.8 GHz {score('58'):.6f} dB。
双频均≤1 dB目标{target}。

输入：256×256 float32 高度图，单位米，每图最小高程为0；TX为整数行列像素。
保持官方高度小数，不要整数化或按高差缩放。
输出：(2,256,256) float32，顺序415 MHz、5.775 GHz，单位正路径损耗dB。
不需要标签、掩码、索引表或特征缓存。

在服务器 CC 目录执行：
~~~bash
{cmd}
~~~
输出位于本包目录。程序拒绝覆盖已有文件。
示例使用 val_index.csv 第一个样本的原始高度图与TX，未读取其标签。
Python接口：src.c10_inference.C10Predictor(model_directory).predict(heightmap,tx_row,tx_col)。

沿用 CC/run_gpu3.sh，以完整UUID限定实体3并复核身份。显存分配上限为总显存15%。
使用CC独立虚拟环境，依赖版本见requirements.lock.txt。
这是服务器复现包，未提交Kaggle或制作最终Notebook。

本机预热后完整双频单图：平均{inf['warm_map_seconds_mean']*1000:.1f}ms，
p95 {inf['warm_map_seconds_p95']*1000:.1f}ms；首次调用{inf['cold_first_map_seconds']*1000:.1f}ms。
包含当场特征计算和输出。官方硬件与计时规则仍需正式核对。
"""
(pkg/'README.md').write_text(readme)
table=[]
for n,title in [('c6_points_summary','几何摘要'),('c6_points_profile','有序剖面'),('c7_points_material','地表穿越与材料表征'),('c7_points_surface','周围可见地面统计'),('c8_points_context','TX邻域与高度小数'),('c9_points_compact','小数条件的临界高度'),('c9_points_combined','组合剖面和周围地面'),('c10_points_phase_geometry','整条剖面的同一高度假设')]:
 d=read(R/'runs'/f'{n}_s20261101'/'result.json');table.append(f"| {title} | {d['bands']['415']['rmse_db']:.6f} | {d['bands']['58']['rmse_db']:.6f} |")
zt=[]
for n,title in [('height_sensitive','高度敏感区'),('clear_near_boundary','可见区边界附近'),('clear_grazing','可见但近掠射'),('clear_robust','稳定可见区'),('shallow_blocked','浅遮挡区'),('deep_blocked','深遮挡区')]:
 a=zones['bands']['415'][n];b=zones['bands']['58'][n];zt.append(f"| {title} | {a['C4_rmse']:.3f} → {a['C10_rmse']:.3f} | {b['C4_rmse']:.3f} → {b['C10_rmse']:.3f} | {a['C10_SSE_fraction']*100:.1f}% / {b['C10_SSE_fraction']*100:.1f}% |")
report=f"""# C10 新版本与完整官方验证（2026-11-01）

官方固定验证RMSE为 {score('415'):.6f} / {score('58'):.6f} dB。
相对C4下降 {gain('415'):.2f}% / {gain('58'):.2f}%。双频均≤1 dB目标{target}。

## 完整官方结果

| 版本 | 415 MHz RMSE（dB） | 5.8 GHz RMSE（dB） |
|---|---:|---:|
| C4 | {base('415'):.6f} | {base('58'):.6f} |
| C10 | {score('415'):.6f} | {score('58'):.6f} |

每频段2,330图，680块地形；有效像素数151,537,682 / 135,926,253。
分频累计全部有效像素的平方误差与像素数，再开方。
NumPy和Torch CPU独立计分，差异均小于1e-8 dB。不是Kaggle格式检查榜分数，本轮没有网站提交。

## 改动与依据

C10以两个独立LightGBM模型预测自由空间参考之外的损耗。
表征来自高度图与TX：距离、相对高度、坡度、有序剖面、临界端点高度、
周围可见面的距离和朝向统计、TX邻域，以及TX高度的小数部分。

代码中的phase是高度的小数部分，不是无线电相位。
将它与临界遮挡高度显式组合，再对整条剖面施加同一高度假设。
没有新增测量或传播仿真标签。几何候选面统计不是实测PAS/PDP。
收益证明现有输入包含旧表示或旧模型没有充分利用的信息。
它不等于恢复了真实TX坐标、原始网格，或证明生成代码取整。

## 同一开发采样点对照

FIT：23,158图×64点；DEV：2,562图×256点；保持原地形划分。
掩码仅进入损失和计分。旧C4在相同DEV采样点为3.757172 / 4.790880 dB。
下表不是官方验证成绩。

| 表征 | 415 MHz（dB） | 5.8 GHz（dB） |
|---|---:|---:|
{chr(10).join(table)}

完整DEV：C9为3.022660 / 3.405284；C10为{ds['bands']['415']['rmse_db']:.6f} / {ds['bands']['58']['rmse_db']:.6f} dB。
材料层表征未带来高频收益。TX/RX半像素对齐在FIT校准/确认中只见很小的类别代理改善，未为此追加正式训练。
负结果和来源均保存在audit与runs。

## 训练与评价

候选与树数由内部开发结果确定。C10固定1,200树/频段。
再拟合使用全部25,720张官方训练地图、6,120块地形；每图64个均匀采样像素，共1,646,080个位置。
分频有效训练点为{fit['bands']['415']['valid_train_sampled_pixels']:,} / {fit['bands']['58']['valid_train_sampled_pixels']:,}。
全部训练地图不表示使用了每图全部像素。

本轮官方验证不进入训练或早停。此前官方验证曾用于误差诊断，不能把此结果称为从未接触的盲测。
仅一个训练种子，不宣称多种子稳定性。

## 剩余误差

同一套输入几何分区；区名不是射线追踪真实状态。

| 分区 | 低频：C4 → C10 | 高频：C4 → C10 | C10平方误差占比：低频 / 高频 |
|---|---:|---:|---:|
{chr(10).join(zt)}

## 复核与交付

只用实体3的完整UUID。GPU进程分配上限15%；完整验证推理峰值{inf['GPU_peak_GB']:.3f}GB。
同输入树预测的CPU/GPU float64最大差小于2.5e-13。
新增GPU高度特征在4张图上的预测差RMSE小于8.7e-5 dB。
独立Heightmap+TX接口与批处理在4张整图上完全一致。
原始18个发布文件大小/mtime及旧C4权重哈希不变。

预热后完整双频单图：平均{inf['warm_map_seconds_mean']*1000:.1f}ms，p95 {inf['warm_map_seconds_p95']*1000:.1f}ms。
首次调用{inf['cold_first_map_seconds']*1000:.1f}ms。测量来自本机共享H100，不替代主办方正式计时。

- 权重与训练记录：{run}
- 完整官方预测与评分：{pred}
- 独立接口：{R}/src/c10_inference.py
- 单图入口：{R}/scripts/c10_predict_one.py
- 服务器复现包：{pkg}
- 分区证据：{R}/audit/c10_official_error_zones.json
"""
rp=R/'results/C10新版本与官方验证_20261101.md';rp.write_text(report)
manifest={str(p.relative_to(pkg)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in pkg.rglob('*') if p.is_file()}
(pkg/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=pkg.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as tf:tf.add(pkg,arcname=pkg.name)
delivery={'version':'C10','official_RMSE_dB':{b:score(b) for b in ['415','58']},'relative_RMSE_reduction_percent':{b:gain(b) for b in ['415','58']},'goal_1dB_both_met':off['goal_met_on_official_validation'],'checks':checks,'report':str(rp),'package':str(pkg),'archive':str(archive),'archive_sha256':sha(archive),'model_sha256':inf['model_sha256'],'website_submission':False}
(R/'audit/c10_final_delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
p=R/'STATUS.md';s=p.read_text();a=s.index('当前工作：');e=s.index('\n\n下文为已完成 C5',a)
current=f"""当前工作：C10新版本已完成训练、完整DEV核验、完整官方固定验证及服务器复现包。
官方固定验证为{score('415'):.6f} / {score('58'):.6f} dB，
较C4下降{gain('415'):.2f}% / {gain('58'):.2f}%。双频均≤1 dB目标{target}。没有网站提交。

全部训练地图每图64个均匀采样像素，固定1,200树/频段；官方验证未进入本轮训练和早停。
独立计分、输入接口、原始数据和旧权重核查通过。
GPU只用实体3完整UUID，推理峰值{inf['GPU_peak_GB']:.2f}GB。
本机预热后完整双频单图平均{inf['warm_map_seconds_mean']*1000:.1f}ms，p95 {inf['warm_map_seconds_p95']*1000:.1f}ms。

交付：results/C10新版本与官方验证_20261101.md、releases/C10_20261101.tar.gz、audit/c10_final_delivery.json。
高度小数与条件几何的收益不等于恢复了真实TX或证明生成流程取整。
剩余误差见audit/c10_official_error_zones.json。"""
p.write_text(s[:a]+current+s[e:]);print(json.dumps(delivery),flush=True)

