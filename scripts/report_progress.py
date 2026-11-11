"""Current facts only, derived from saved run histories and independent scores."""
import json,csv,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];records=[]
for run in sorted((ROOT/'runs').iterdir()):
 if not (run/'config.json').exists():continue
 cfg=json.loads((run/'config.json').read_text());history=json.loads((run/'history.json').read_text()) if (run/'history.json').exists() else []
 progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {'status':'starting'}
 best=min(history,key=lambda x:x['mse_mean']) if history and cfg['part']=='fit' else None
 if cfg['part']=='full':
  score_path=run/'predictions/val_full_bf16/independent_score.json'
  if score_path.exists():
   official=json.loads(score_path.read_text());best={'epoch':cfg['fixed_stop_epoch'],'rmse415':official['bands']['415']['rmse_db'],'rmse58':official['bands']['58']['rmse_db']}
 records.append({'run':run.name,'mode':cfg['mode'],'part':cfg['part'],'seed':cfg['seed'],'status':progress['status'],'completed_epochs':len(history),'selected_epoch':best['epoch'] if best else None,'selected_rmse415_db':best['rmse415'] if best else None,'selected_rmse58_db':best['rmse58'] if best else None,'stable_train_maps_per_second_median':__import__('statistics').median([x['training_maps_per_second'] for x in history[1:]]) if len(history)>1 else None})
lines=['# CC 训练进展','',f'更新时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}','', '目标：官方固定验证集的两个频段 RMSE 分别不高于 1 dB。下表中的开发集结果不代表官方目标达成。','', '| 运行 | 评价划分 | 状态 | 完成轮数 | 同一检查点轮数 | 415 MHz RMSE / dB | 5.8 GHz RMSE / dB |','|---|---|---|---:|---:|---:|---:|']
for r in records:
 f=lambda v:'—' if v is None else f'{v:.4f}'
 lines.append(f"| {r['run']} | {'开发验证' if r['part']=='fit' else '官方验证'} | {r['status']} | {r['completed_epochs']} | {r['selected_epoch'] or '—'} | {f(r['selected_rmse415_db'])} | {f(r['selected_rmse58_db'])} |")
lines+=['','表内两个频段始终来自同一个检查点。开发检查点按开发集双频等权 MSE 选择；它不是官方归一化总分。完整训练使用开发集固定的停止轮数，官方验证不参与训练轮数选择。','', '所有 GPU 任务只允许实体 3 号 H100，UUID：GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0。','', '独立复算：']
for run in sorted((ROOT/'runs').iterdir()):
 for p in sorted(run.glob('predictions/*/independent_score.json')):
  s=json.loads(p.read_text());lines.append(f"- {p.relative_to(ROOT)}：{s['evaluation_split']}，415 MHz {s['bands']['415']['rmse_db']:.6f} dB，5.8 GHz {s['bands']['58']['rmse_db']:.6f} dB。")
(ROOT/'results/训练进展.md').write_text('\n'.join(lines)+'\n');(ROOT/'results/run_summary.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False))
