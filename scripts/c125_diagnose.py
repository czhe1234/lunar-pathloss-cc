"""Saved-prediction analysis. All oracle interventions are diagnostics, never deployable results."""
import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c125_tail';S=R/'audit/c101_residual/monitor'
t=np.load(S/'target.npy',mmap_mode='r');base=np.load(S/'base.npy',mmap_mode='r');tail_union=np.load(R/'audit/c120_sign/monitor_rows.npy')
out={'scope':'same C101 TRAIN monitor, 128 terrains; no new fitting','bands':[]}
for b in range(2):
 sel=np.flatnonzero(t[:,b+2]>0);res=(t[sel,b].astype('float64')-base[sel,b]);tail=np.abs(res)>=5
 savedrows=np.load(R/f'runs/c124_spectrum/monitor_rows{b}.npy');assert np.array_equal(sel,savedrows)
 d=np.load(R/f'runs/c124_spectrum/monitor_delta{b}.npy').astype('float64');r=res[tail];z=d[tail]
 def rmse(v):return float(np.sqrt(np.mean((res-v)**2)))
 oracle_sign=d.copy();oracle_sign[tail]=np.sign(r)*np.abs(z)
 oracle_magnitude=d.copy();oracle_magnitude[tail]=np.where(z>=0,1.,-1.)*np.abs(r)
 oracle_both=d.copy();oracle_both[tail]=r
 oracle_gate=np.where(tail,d,0)
 pos=r>0;true_sign=np.where(pos,1.,-1.);correct=(z>=0)==pos
 rec=dict(band=b,valid=len(sel),tail=int(tail.sum()),tail_positive_fraction=float(pos.mean()),majority_accuracy=float(max(pos.mean(),1-pos.mean())),
 C82_RMSE=rmse(np.zeros_like(d)),C124_RMSE=rmse(d),
 conditional=dict(sign_accuracy=float(correct.mean()),balanced_sign_accuracy=float(.5*(correct[pos].mean()+correct[~pos].mean())),squared_residual_weighted_sign_accuracy=float(np.sum(correct*r*r)/np.sum(r*r)),positive_recall=float(correct[pos].mean()),negative_recall=float(correct[~pos].mean()),target_abs_q50_q90_q99=np.quantile(np.abs(r),[.5,.9,.99]).tolist(),predicted_abs_q50_q90_q99=np.quantile(np.abs(z),[.5,.9,.99]).tolist(),mean_residual_positive=float(r[pos].mean()),mean_residual_negative=float(r[~pos].mean())),
 truth_assisted_diagnostics=dict(correct_sign_only=rmse(oracle_sign),correct_magnitude_only=rmse(oracle_magnitude),perfect_tail_correction=rmse(oracle_both),perfect_tail_selection_only=rmse(oracle_gate)))
 expected=sel[tail];keep=(t[tail_union,b+2]>0)&(np.abs(t[tail_union,b]-base[tail_union,b])>=5);assert np.array_equal(expected,tail_union[keep])
 for mode in ['old','native']:
  prob=np.load(R/f'runs/c120_{mode}/monitor_prob{b}.npy').astype('float64');ok=(prob>=.5)==pos
  rec['C120_'+mode]=dict(conditional_sign_accuracy=float(ok.mean()),balanced_accuracy=float(.5*(ok[pos].mean()+ok[~pos].mean())),squared_residual_weighted_accuracy=float(np.sum(ok*r*r)/np.sum(r*r)),scope='trained and evaluated conditional on true tail; not all-point deployment')
 out['bands'].append(rec)
(A/'diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
