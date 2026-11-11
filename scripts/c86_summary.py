"""Collect the final evidence, without changing the model or choosing on official labels."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver'
def read(p):return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
delivery=read(A/'official_delivery.json');baseline=read(R/'audit/c76_final_delivery.json')['official']
version=read(R/'submissions/C82_Kaggle_20261103/manifest.json');release=R/'releases/C82_20261103.tar.gz'
csv=R/'submissions/C82_Kaggle_20261103/submission_C82.csv'
assert version['rows']==800000 and version['sha256']==digest(csv)
assert delivery['archive_sha256']==digest(release) and not delivery['website_submission']
new=delivery['official']['RMSE'];assert all(b>a for b,a in zip(baseline['RMSE'],new)) and not delivery['both_major'] and not delivery['target']
replay=read(A/'package_replay_gpu3.json')
rdata=read(A/'rt_score.json');sdata=read(A/'angular_score.json')
v79=read(A/'confirm_score.json')['results']['combined']
solver=read(A/'targeted_sionna.json');assert solver['full_4M_vs_sparse4M']['max_abs_dB']<1e-4
assert solver['sparse4B_pad_065_vs_135']['max_abs_dB']<1e-4 and solver['terrains']==39
budget=read(A/'c82_error_budget.json');np.testing.assert_allclose(budget['official_RMSE'],new,atol=1e-5)
c84=read(R/'audit/c84_tx/decision.json')
assert c84['maps']==1934 and not c84['promotion']
out=dict(winner='C82',major_delivered=False,both_official_le_1=False,official_frozen_baseline=baseline,
    actual_C82=delivery,official_terrain_paired=read(A/'c82_official_paired.json'),package_timing_replay=replay,
    submission=version,geometry_screen_confirm484=v79,finite_original_solver_4M=dict(calibration_maps=rdata['fit_maps'],confirm_maps=rdata['n']['confirm'],confirm=rdata['results']['confirm']['full']),
    angular_geometry_confirm=sdata['results']['confirm']['full'],weak_state_training=read(A/'c81_paired.json'),
    multiscale_fusion_internal=read(A/'blend_decision.json'),transmitter_transfer=read(A/'terrain_cross_tx.json'),paired_transmitter_training=c84,
    full_map_oracle_budget=budget,known_scene_primary_positive_control=read(A/'known_scene_control.json'),
    targeted_4B_original_solver=solver,targeted_power_check=read(A/'targeted_amplitude.json'),examples=read(A/'matched_counterexamples.json'),
    score_scope='Published fixed validation, original masks, all 2330 maps/band on 680 terrains; not hidden test or Kaggle website.',
    research_scope='Only official TRAIN for new model selection/training and for the Sionna positive controls. Official validation only after freezing for C82 scoring and descriptive error budget.',
    runtime_cache=dict(initial_library_default=True,own_entries_moved_to_CC=160,scope='Only runtime LLVM compiler entries; subsequent runs isolated using bwrap.',record='runtime_cache_relocated.json'),
    source='CC independent code plus official Sionna RT 1.1.0 for known scene controls. GPU work restricted to authorized UUIDs 1,3,4. No new neural training as part of C85/C86.',
    decision='Close local C76 residual corrections as the route to 1 dB; none of the new training branches met the major-effect threshold. Weak receiver location must transfer between intentionally perturbed and nominal physical scenes on unseen TRAIN terrain before another large run.',
    evidence_root=str(A),release_path=str(release),csv_path=str(csv))
(A/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
p=R/'STATUS.md';s=p.read_text();head='## 2026-11-03 C84 与接收弱点定位复核'
if head not in s:
    p.write_text(s.rstrip()+'\n\n'+head+'\n\n相同权重和更新预算下，保留旧发射位置的监测得分最优。484 块确认地形上该组仅改善 0.0044 / 0.0070 dB，结束扩训。在已知几何的模拟上，独立计算能够定位首个接收面直达命中。原版 Sionna 按标称 40 亿发射方向复核比赛 TRAIN 的 39 块地形、1,126 个风险位置，正确发射方向不能稳定优于转偏对照。结合完整官方误差分解，单独纠正 4×4 块均值仍剩 1.894 / 2.120 dB。当前可提交版本 C82 综合为 2.412764 dB。报告见 results/C82官方得分与方法复核_20261103.md。\n')
print('stored',A/'summary.json',(A/'summary.json').stat().st_size)
