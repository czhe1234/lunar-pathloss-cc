"""Record the completed C12 decision without changing the delivered C10 model."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
d=json.loads((R/'audit/c12_decision.json').read_text())
assert not d['promote_to_full_validation'], 'Successful C12 requires implementation and full validation, not this negative-result recorder.'
a=d['bands']['415'];b=d['bands']['58']
report=f"""# C12 一致高度假设下的地表表征检验

问题：C10的高度修正主要进入直线剖面；将同一修正也用于周围地表可见性和地表穿越，能否继续明显降低误差？

在C10的294维输入后，增加20维地表几何统计、16维穿越/材料描述。
这些特征只由原始高度图与TX推导；没有仿真新标签，也没有恢复实际TX。
使用原CC实现的参数化版本，TX3条件下在4张训练图上与原特征逐元素一致。

训练和评价使用与C10相同的FIT/DEV地形划分、均匀像素、标签、掩码和参数。
每频段最多1,200棵树；种子20261101。没有用官方验证拟合、早停或选择C12。

| 版本 | 415 MHz开发采样RMSE | 5.8 GHz开发采样RMSE |
|---|---:|---:|
| C10 | {a['C10_RMSE_db']:.6f} | {b['C10_RMSE_db']:.6f} |
| C12 | {a['C12_RMSE_db']:.6f} | {b['C12_RMSE_db']:.6f} |
| 相对下降 | {a['relative_reduction_percent']:.2f}% | {b['relative_reduction_percent']:.2f}% |

这是固定开发采样点结果，不是完整官方验证成绩。
C12训练采样RMSE为 {a['C12_fit_RMSE_db']:.6f} / {b['C12_fit_RMSE_db']:.6f} dB。

**决定：不进入全训练再拟合和完整官方验证。**
训练前已规定两个频段都至少改善3%才继续；此次未通过。
保留C10为已完成完整官方验证和交付复核的版本。
此结果只检验这组特征与当前模型，不把小幅收益写成解决了剩余误差。

协议：audit/c12_protocol.json。
特征复核：audit/c12_feature_replay_check.json。
训练记录：runs/c12_points_consistent_height_s20261101。
决定：audit/c12_decision.json。

全部工作在CC内完成。此分支使用CPU；其它GPU任务没有被停止或改动。
"""
p=R/'results/C12地表表征检验_20261101.md';assert not p.exists();p.write_text(report)
s=(R/'STATUS.md').read_text();old='\n\n后续受控试验C12进行中：仅在内部FIT/DEV检验同一高度修正下的地表可见性与地表穿越；\n事先规定两个频段均相对C10改善至少3%才继续完整验证。协议：audit/c12_protocol.json。'
assert old in s
new=f'\n\n后续受控试验C12已完成：开发采样RMSE {a["C12_RMSE_db"]:.6f} / {b["C12_RMSE_db"]:.6f} dB，\n相对C10下降 {a["relative_reduction_percent"]:.2f}% / {b["relative_reduction_percent"]:.2f}%，未通过事先规定的双频各3%门槛。\n该分支没有进入全训练再拟合或官方验证。当前交付版本仍是C10。详见results/C12地表表征检验_20261101.md。'
(R/'STATUS.md').write_text(s.replace(old,new,1))
p10=R/'results/C10新版本与官方验证_20261101.md';s=p10.read_text();s+=f'\n## 后续受控检验\n\nC12把同一高度假设扩展到地表可见性与穿越。\n开发采样RMSE为{a["C12_RMSE_db"]:.6f} / {b["C12_RMSE_db"]:.6f} dB，\n相对C10改善{a["relative_reduction_percent"]:.2f}% / {b["relative_reduction_percent"]:.2f}%，没有通过预设的双频各3%门槛。\n没有为此追加完整再拟合和官方验证。详见{p}。\n';p10.write_text(s)
q=R/'audit/c10_final_delivery.json';v=json.loads(q.read_text());v['subsequent_controlled_experiment']={'name':'C12','decision':str(R/'audit/c12_decision.json'),'report':str(p),'promoted':False};q.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'report':str(p),'C12_promoted':False,'C10_release_preserved':True}))
