"""Add independently recomputed residual evidence to the C10 report."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
d=read(R/'audit/c10_official_remaining_diagnostic.json');old=read(R/'audit/c4_remaining_error_20261031/joint_band_audit.json')['results']['all'];qa=read(R/'audit/c10_package_verification.json')
assert qa['passed'] and old['joint_valid_pixels']==d['joint_valid_pixels']
a=d['bands']['415'];b=d['bands']['58']
section=f"""## 进一步定位剩余误差

全部官方验证上，平均偏差只有 {a['mean_error_db']:.4f} / {b['mean_error_db']:.4f} dB。
这排除了用一个全局常数偏移消除主要误差的办法。

绝对误差>10 dB的像素占 {100*a['tails']['10']['pixel_fraction']:.2f}% / {100*b['tails']['10']['pixel_fraction']:.2f}%，
贡献 {100*a['tails']['10']['SSE_fraction']:.2f}% / {100*b['tails']['10']['SSE_fraction']:.2f}% 的平方误差。
即使理想地修好全部这些像素、其它预测不变，全体RMSE仍为
{a['RMSE_if_all_errors_over_10db_were_perfectly_removed']:.3f} / {b['RMSE_if_all_errors_over_10db_were_perfectly_removed']:.3f} dB。
这是误差账目，不是可实现模型或误差下限。

按附加损耗<3、3–10、≥10 dB分区：
标签和预测都≥10 dB时，仍贡献 {100*a['both_high_loss_SSE_fraction']:.2f}% / {100*b['both_high_loss_SSE_fraction']:.2f}% 平方误差，
区内RMSE为 {a['both_high_loss_RMSE_db']:.3f} / {b['both_high_loss_RMSE_db']:.3f} dB。
这说明只修“是否属于高衰减”的分类还不够，区内损耗强度也必须预测准确。
这里的损耗分区不是真实逐径传播状态。

在两个频段共同有效的 {d['joint_valid_pixels']:,} 个像素上，
双频差的预测误差RMSE从C4的 {old['band_difference_error_rmse']:.3f} 降到C10的
{d['dual_band_difference_error_RMSE_db']:.3f} dB。
共同残差能量占比为 {100*d['common_residual_energy_fraction']:.2f}%，这是代数分解，不是地形原因份额。

将残差投影为8×8块内均值，保留 {100*a['8x8_residual_mean_projection_SSE_fraction']:.2f}% /
{100*b['8x8_residual_mean_projection_SSE_fraction']:.2f}% 平方误差。
该结果不支持把剩余误差全当成整图偏移，也不证明细尺度残差不可学习。

要达到双频1 dB，还需消除当前各频段约 {100*(1-1/a['RMSE_db']**2):.2f}% /
{100*(1-1/b['RMSE_db']**2):.2f}% 的平方误差。
这不是靠当前幅度的小修正就已接近的目标。

证据：{R}/audit/c10_official_remaining_diagnostic.json。
服务器复现包已执行其自己的单图CLI，双频输出与完整官方预测逐像素一致；
19个清单文件与压缩包哈希也已核对。证据：{R}/audit/c10_package_verification.json。

"""
p=R/'results/C10新版本与官方验证_20261101.md';s=p.read_text();assert '## 进一步定位剩余误差' not in s
s=s.replace('## 复核与交付',section+'## 复核与交付');p.write_text(s)
status=R/'STATUS.md';s=status.read_text();tag='\n\n下文为已完成 C5';assert tag in s
s=s.replace(tag,'\n\n后续受控试验C12进行中：仅在内部FIT/DEV检验同一高度修正下的地表可见性与地表穿越；\n事先规定两个频段均相对C10改善至少3%才继续完整验证。协议：audit/c12_protocol.json。'+tag,1);status.write_text(s)
print(json.dumps({'report_updated':str(p),'package_verified':True,'remaining_RMSE_db':[a['RMSE_db'],b['RMSE_db']]}))
