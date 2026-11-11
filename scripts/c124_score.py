"""Independent official-formula masked-RMSE calculation on fixed TRAIN holdout points."""
from pathlib import Path
import numpy as np,json,csv
R=Path(__file__).resolve().parents[1];A=R/'audit/c124_spectrum';S=R/'audit/c101_residual/monitor'
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
mapid=np.load(S/'mapid.npy',mmap_mode='r')
terrain=np.array([int(rows[int(i)]['terrain_id']) for i in mapid],dtype='int32')
unique,group=np.unique(terrain,return_inverse=True)
target=np.load(S/'target.npy',mmap_mode='r');base=np.load(S/'base.npy',mmap_mode='r')
rng=np.random.default_rng(124083);boot=rng.integers(0,len(unique),size=(4000,len(unique)))
out={'scope':'C101 sampled native pixels on 128 TRAIN held-out terrains; not official VAL','terrains':len(unique),'bands':[]}
for b in range(2):
 sel=np.flatnonzero(target[:,b+2]>0);g=group[sel];y=target[sel,b].astype('float64');p=base[sel,b].astype('float64');res=y-p
 n=np.bincount(g,minlength=len(unique));s0=np.bincount(g,weights=(p-y)**2,minlength=len(unique))
 counts=float(n.sum());base_rmse=float(np.sqrt(s0.sum()/counts));record={'band':b,'n':len(sel),'C82':base_rmse}
 estimates={}
 for mode in ['total','spectrum']:
  run=R/f'runs/c124_{mode}';assert np.array_equal(sel,np.load(run/f'monitor_rows{b}.npy'))
  delta=np.load(run/f'monitor_delta{b}.npy').astype('float64');pred=p+delta
  sse=np.bincount(g,weights=(pred-y)**2,minlength=len(unique));rmse=float(np.sqrt(sse.sum()/counts))
  gains=np.sqrt(s0[boot].sum(1)/n[boot].sum(1))-np.sqrt(sse[boot].sum(1)/n[boot].sum(1))
  saved=json.load(open(run/f'result{b}.json'));assert abs(rmse-saved['monitor_rmse'])<1e-6
  tail=np.abs(res)>=5
  record[mode]=dict(rmse=rmse,gain_vs_C82=base_rmse-rmse,gain_CI95=np.quantile(gains,[.025,.975]).tolist(),tail_n=int(tail.sum()),tail_rmse=float(np.sqrt(np.mean((pred[tail]-y[tail])**2))),tail_sign_accuracy=float(np.mean((delta[tail]>=0)==(res[tail]>=0))),delta_rms=float(np.sqrt(np.mean(delta**2))),trees=saved['trees'])
  estimates[mode]=sse
  np.save(A/f'{mode}_terrain_sse{b}.npy',sse)
 diff=np.sqrt(estimates['total'][boot].sum(1)/n[boot].sum(1))-np.sqrt(estimates['spectrum'][boot].sum(1)/n[boot].sum(1))
 record['spectrum_gain_vs_total']=record['total']['rmse']-record['spectrum']['rmse'];record['spectrum_gain_vs_total_CI95']=np.quantile(diff,[.025,.975]).tolist()
 record['spectrum']['tail_pixel_fraction']=record['spectrum']['tail_n']/record['n']
 record['spectrum']['tail_sse_fraction']=record['spectrum']['tail_n']*record['spectrum']['tail_rmse']**2/(record['n']*record['spectrum']['rmse']**2)
 out['bands'].append(record)
out['monitor_gate_passed']=all(x['spectrum']['gain_vs_C82']>=.20 for x in out['bands'])
out['official_val_used']=False
(A/'score.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
