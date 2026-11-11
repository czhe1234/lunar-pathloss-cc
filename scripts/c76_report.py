"""Publish measurements and hash-check the release after normal input-only replay."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c72_continuation';run=R/'runs/c76_selected';release=R/'releases/C76_20261103'
def get(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pair(x):return ' / '.join(f'{v:.4f}' for v in x)
def main():
    replay=get(A/'package_replay.json');assert replay['verified']
    off=get(R/'audit/c76_selected_official/verified_summary.json');old=get(R/'audit/c65_combo_full_official/verified_summary.json')
    fr=get(run/'freeze.json');cfg=get(run/'model.json');scores=get(fr['mix_source'])
    curr=np.array(off['RMSE']);gain=np.array(old['RMSE'])-curr
    assert (gain>0).all()
    csv=get(R/'submissions/C76_Kaggle_20261103/manifest.json');assert csv['all_output_values_match_float32_inference']
    bench={k:get(A/f'full_{k}/result.json') for k in ['H100','A800']};qa=get(A/'final_raw_interface/result.json')
    lines=['# C76 双频路损预测实验记录','',
        f"官方固定验证的综合 RMSE 从 {old['combined_rmse_db_masked']:.6f} 降至 {off['combined_rmse_db_masked']:.6f} dB。",
        '', '| 正式版本 | 415 MHz RMSE | 5.775 GHz RMSE | 综合 RMSE |','|---|---:|---:|---:|']
    for k,v in [('C65',old),('C76',off)]:lines.append(f"| {k} | {v['RMSE'][0]:.6f} | {v['RMSE'][1]:.6f} | {v['combined_rmse_db_masked']:.6f} |")
    lines.extend(['',f"两个频段分别改善 {gain[0]:.6f} 和 {gain[1]:.6f} dB。"
        +('两频段都达到 1 dB 目标。' if np.all(curr<=1) else '双频均不高于 1 dB 的目标尚未达到。'),
        '',f"官方固定验证覆盖 680 块地形和每频段 2,330 张图。有效像素数分别为 {off['valid_pixels'][0]:,} 和 {off['valid_pixels'][1]:,}。先按频段累计有效像素平方误差，再除以有效像素数并开方。综合值取两频段 RMSE 的算术平均。原版官方脚本与独立 float64 复算的差小于 0.00001 dB。",
        '', '## 同口径内部验证', '', '下表为官方 TRAIN 内的 612 块地形，共 2,562 张完整图。单位均为 dB。','',
        '| 模型 | 415 MHz | 5.775 GHz |','|---|---:|---:|'])
    models=[('C54 内部基线',scores['all612']['C54'])]
    for label,n in [('C69 传播特征','c69_registered_spatial'),('C70 局部几何','c70_registered_geometry_resume'),('C70 多尺度','c70_corridor_field_resume'),('C72 先预测接收单元','c72_native'),('C72 先合并接收单元','c72_observed'),('C75 扩大局部训练','c75_wide_observed')]:
        p=R/'runs'/n/'full_internal/result.json'
        if p.exists():models.append((label,get(p)['all612']['new']))
    for name,k in [('C73 真实地表连接','real_graph'),('C73 打乱地表连接','shuffled_graph'),('冻结组合','chosen')]:models.append((name,scores['all612'][k]))
    for key,v in models:lines.append(f'| {key} | {v[0]:.6f} | {v[1]:.6f} |')
    o=scores['outside484'];lines.extend(['',f"在原选轮数面板以外的 484 块内部地形上，冻结组合为 {pair(o['chosen'])} dB，相对 C54 改善 {pair(o['gain'])} dB。95% 区间按地形配对重采样。",
        '','## 训练与模型选择','',
        'C70 几何版从第 14 轮恢复，多尺度版从第 11 轮恢复，均重新初始化了优化器。两组分别完成第 16 和第 24 轮。C72 的两组使用相同的 1,279 张 FIT 图，比较接收三角单元先提取非线性特征或先插值的效果。','',
        'C73 在全部 5,508 块内部 FIT 地形中，每块至多选 8 个发射位置，共 7,979 张图。真实连接组沿可见地表完成两次特征传递。打乱组保持网络、参数量、初始权重、训练次序和预算，只替换可见边所指向的源节点。','',
        'C75 从 C72 最佳权重继续训练。增加 960 块局部模型此前未拟合的 FIT 地形和 1,536 张图，合计使用 2,815 张图。训练覆盖与更新次数同时增加。','',
        f"冻结配置选用 {cfg.get('internal_local_run')}。全官方 TRAIN 拟合的 C65 提供基础预测。图模型的共享系数为 {cfg['graph_weight']:.3f}。"
        +(f" 分区权重已启用，两个频段的四类系数分别为 {cfg['gate_weights']}。" if cfg.get('gate_weights') is not None else '所有像素使用同一组合规则。'),
        '', '接收三角网格与向下一行读取发射高度均为基于训练数据验证的候选几何，原始仿真网格与实际发射机三维坐标仍未获得。候选、训练轮数和组合系数均在官方 TRAIN 内确定。官方 VAL 用于冻结后的计分和版本对比。网站成绩与隐藏测试另行产生。',
        '', '## 运行时间与交付',''])
    for k,v in bench.items():lines.append(f"{k} 在 12 张不同 TRAIN 图上共计 {v['warm_calls']} 次预热双频调用，平均 {1000*v['warm_mean']:.1f} ms，95 分位 {1000*v['warm_p95']:.1f} ms，最慢 {1000*v['warm_max']:.1f} ms。首次调用含编译，为 {v['cold']:.1f} 秒。\n")
    lines.extend(['以上计时覆盖原始高度图、TX 输入到两张完整路损图。比赛要求的推理时间上限为 500 ms。[比赛说明](https://lunarradiomapchallenge.github.io/)',
        '', f"提交 CSV 按官方模板的 {csv['rows']:,} 个 ID 导出。每个数值均与冻结模型的 float32 预测逐一核对。服务器复现包的单图入口已重放验证。"])
    timing=[]
    for folder in sorted((run/'predictions').glob('val_full_c34_fresh_shard*')):
        vals=np.load(folder/'seconds_per_map.npy');meta=get(folder/'inference.json')
        assert len(vals)==meta['maps'] and np.isfinite(vals).all()
        warm=vals[1:] if meta['resumed_at']==0 else vals
        timing.append(dict(gpu=meta['gpu']['name'],uuid=meta['gpu']['uuid'],maps=len(vals),warm_maps=len(warm),warm_over_limit=int((warm>.5).sum()),warm_mean=float(warm.mean()),warm_p95=float(np.quantile(warm,.95)),warm_max=float(warm.max())))
    lines.extend(['','正式验证逐图保存了端到端耗时。以下统计分别列出每个分片的预热后调用。'])
    for v in timing:lines.append(f"{v['gpu']} 分片共 {v['warm_maps']} 张，超过 500 ms 的有 {v['warm_over_limit']} 张，最慢 {v['warm_max']*1000:.1f} ms。")
    # Terrain resampling keeps all transmitters of a held-out terrain together.
    import csv as csv_module
    rows=list(csv_module.DictReader((R.parent/'data/val_index.csv').open()))
    bdata=np.load(R/'runs/c65_combo_full/per_map_val.npz');ndata=np.load(run/'per_map_val.npz')
    ib=np.argsort(bdata['ids']);inew=np.argsort(ndata['ids']);assert np.array_equal(bdata['ids'][ib],ndata['ids'][inew])
    assert np.array_equal(bdata['count'][ib],ndata['count'][inew])
    terrain=np.asarray([rows[int(i)]['terrain_id'] for i in ndata['ids'][inew]]);names,inv=np.unique(terrain,return_inverse=True)
    per=np.zeros((len(names),3,2));np.add.at(per,inv,np.stack((bdata['sse'][ib],ndata['sse'][inew],ndata['count'][inew]),1))
    rng=np.random.default_rng(76003);samp=per[rng.integers(len(names),size=(3000,len(names)))].sum(1)
    gain_draw=np.sqrt(samp[:,0]/samp[:,2])-np.sqrt(samp[:,1]/samp[:,2]);ci=np.quantile(gain_draw,[.025,.975],axis=0).T.tolist()
    win=(per[:,1]<per[:,0]);paired=dict(terrains=len(names),paired_delta95=ci,terrains_with_lower_error=win.sum(0).tolist(),jointly_lower=int(win.all(1).sum()))
    lines.extend(['','按官方验证地形配对重采样 3,000 次。低频改善的 95% 区间为 '
        +f"[{ci[0][0]:.4f}, {ci[0][1]:.4f}] dB，高频为 [{ci[1][0]:.4f}, {ci[1][1]:.4f}] dB。"
        +f" 在 680 块地形中，有 {paired['jointly_lower']} 块同时改善两个频段。"])
    repeated=A/'latency_repeat.json'
    if repeated.exists():
        for rec in get(repeated)['records']:
            lines.append(f"原卡复测样本 {rec['sample_id']}。原记录 {1000*rec['original']:.1f} ms，复测 {len(rec['repeat'])} 次，平均 {1000*rec['mean']:.1f} ms，最慢 {1000*rec['max']:.1f} ms。上面的正式验证统计保留原记录。")
    zones=get(A/'c76_internal_zones.json')['zones'];clear=zones['clear'];weak=zones['shadow_weak_first_order']
    isolated=np.asarray(clear['rmse'])*np.sqrt(clear['valid_share'])
    lines.extend(['','## 距离 1 dB 的差距','',f"在 612 块内部地形上，高频的候选深遮挡、弱一次传播区域占有效像素 {100*weak['valid_share'][1]:.1f}%，贡献 {100*weak['sse_share'][1]:.1f}% 的平方误差。全可见候选区贡献了低频 {100*clear['sse_share'][0]:.1f}% 和高频 {100*clear['sse_share'][1]:.1f}% 的平方误差。只计这些全可见区的现有平方误差，用全部有效像素数作分母，仍为 {pair(isolated)} dB。达到双频 1 dB 还需要降低全可见区误差。这些分区来自输入计算，实际传播路径未观测。"])
    probe=A/'c77_offset_result.json'
    if probe.exists():
        d=get(probe);lines.extend(['','## 接收面数值偏移对照','',f"用 64 块 FIT 地形校准，在另 64 块内部地形上确认。候选修正带来的双频 RMSE 改善为 {pair(d['gain'])} dB。该结果只覆盖所测候选网格与求积密度。"])
    report=R/'results/C72-C76研究与官方验证_20261103.md';report.write_text('\n'.join(lines)+'\n')
    archive=release.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=5) as tf:tf.add(release,arcname=release.name,filter=lambda t:None if '__pycache__' in t.name or t.name.endswith('.pyc') else t)
    with tarfile.open(archive) as tf:
        for rel,digest in fr['files'].items():assert hashlib.sha256(tf.extractfile(release.name+'/'+rel).read()).hexdigest()==digest
    out=dict(model='C76',official=off,prior_C65=old['RMSE'],gains=gain.tolist(),goal_both_le_1=bool(np.all(curr<=1)),large_both_ge_point3=bool(np.all(gain>=.3)),
        model_config=cfg,paired_official=paired,full_validation_timing=timing,release_replay=replay,benchmarking={k:{t:v[t] for t in ['warm_mean','warm_p95','warm_max','cold','maps','warm_calls']} for k,v in bench.items()},
        internal_QA_worst_cached_RMS_db=max(max(x['cached_replay_RMS_diff']) for x in qa['records']),report=str(report),submission=csv,
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),website_submission=False)
    (R/'audit/c76_final_delivery.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    status=R/'STATUS.md';tag='## 2026-11-03 C76 官方固定验证'
    if tag not in status.read_text():
        with status.open('a') as f:f.write(f"\n\n{tag}\n\n{lines[2]} 分频为 {pair(curr)} dB。完整记录见 results/{report.name}。服务器复现包和提交 CSV 均已核对。\n")
    print(json.dumps({'RMSE':curr.tolist(),'combined':off['combined_rmse_db_masked'],'gain':gain.tolist(),'archive':str(archive),'CSV':csv['sha256']}),flush=True)
if __name__=='__main__':main()
