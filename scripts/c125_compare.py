"""Independent paired-terrain scoring of the fixed-index neural controls."""
from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c125_tail';P=R/'audit/c101_residual/monitor'
target=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r');ids=np.load(R/'audit/c120_sign/monitor_rows.npy')
maps=np.load(P/'mapid.npy',mmap_mode='r');csvrows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));terrain=np.array([int(csvrows[int(i)]['terrain_id']) for i in maps])
unique,group=np.unique(terrain,return_inverse=True);nterrain=len(unique);boot=np.random.default_rng(125301).integers(nterrain,size=(4000,nterrain))
pred={};prob={};configs={};results={}
for mode in ['old','raw','shuffled']:
 run=R/f'runs/c125_nn_{mode}_fixed';configs[mode]=json.load(open(run/'config.json'));results[mode]=json.load(open(run/'result.json'));assert np.array_equal(np.load(run/'monitor_rows.npy'),ids)
 pred[mode]=np.load(run/'monitor_delta.npy').astype('float64');prob[mode]=np.load(run/'monitor_prob.npy').astype('float64')
assert len({c['initial_sha256'] for c in configs.values()})==1
assert len({c['sampling_sha256'] for c in configs.values()})==1
assert len({tuple(v['fit_panel']['n']) for v in results.values()})==1
out={'scope':'same 128 terrain TRAIN monitor; conditional true-tail experiments, not deployment','bands':[]}
for b in range(2):
 res=target[:,b].astype('float64')-base[:,b];valid=target[:,b+2]>0;tail=(np.abs(res)>=5)&valid
 n=np.bincount(group[valid],minlength=nterrain);tailg=group[tail];nt=np.bincount(tailg,minlength=nterrain);outside=np.bincount(group[valid&~tail],weights=res[valid&~tail]**2,minlength=nterrain)
 idx_tail=np.flatnonzero(tail);keep=(target[ids,b+2]>0)&(np.abs(res[ids])>=5);assert np.array_equal(ids[keep],idx_tail)
 rr=res[tail];g=group[tail];record={'band':b,'tail':int(tail.sum()),'C82_tail_rmse':float(np.sqrt(np.mean(rr**2)))};sums={}
 for mode in ['old','raw','shuffled']:
  d=pred[mode][keep,b];pr=prob[mode][keep,b];s=np.bincount(g,weights=(d-rr)**2,minlength=nterrain);sums[mode]=s
  rmse=float(np.sqrt(s.sum()/nt.sum()));assert abs(rmse-results[mode]['monitor']['tail_RMSE'][b])<1e-6
  record[mode]=dict(tail_RMSE=rmse,sign_accuracy=float(np.mean((d>=0)==(rr>=0))),classification_accuracy=float(np.mean((pr>=.5)==(rr>=0))),truth_gated_all_RMSE=float(np.sqrt((outside+s).sum()/n.sum())))
 for comparator in ['old','shuffled']:
  gain=np.sqrt((outside[boot]+sums[comparator][boot]).sum(1)/n[boot].sum(1))-np.sqrt((outside[boot]+sums['raw'][boot]).sum(1)/n[boot].sum(1))
  record['raw_vs_'+comparator]=dict(truth_gated_RMSE_gain=record[comparator]['truth_gated_all_RMSE']-record['raw']['truth_gated_all_RMSE'],CI95=np.quantile(gain,[.025,.975]).tolist())
 out['bands'].append(record)
(A/'neural_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
