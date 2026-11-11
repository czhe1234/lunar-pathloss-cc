"""Frozen deployable mixture vs truth-assisted membership, no fitting or threshold search."""
from pathlib import Path
import numpy as np,json
from catboost import CatBoostRegressor
R=Path(__file__).resolve().parents[1];A=R/'audit/c125_tail';S=R/'audit/c101_residual'
P=S/'monitor';target=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r')
gate=np.load(R/'audit/c118_location/location_monitor_probability.npy',mmap_mode='r')
assert gate.shape==(len(target),2,3) and np.max(np.abs(gate.sum(2)-1))<1e-5
out={'scope':'fixed C101 TRAIN monitor; frozen C118 probability and C125 old169 conditional mean; no fit or threshold selection','bands':[]}
for b in range(2):
 T=S/'fit';t=np.load(T/'target.npy',mmap_mode='r');bb=np.load(T/'base.npy',mmap_mode='r');w=np.load(T/'weight.npy',mmap_mode='r')
 rr=t[:,b].astype('float64')-bb[:,b];non=(t[:,b+2]>0)&(abs(rr)<5);mu0=float(np.sum(rr[non]*w[non])/np.sum(w[non]))
 model=CatBoostRegressor();model.load_model(str(R/f'runs/c125_tree_old/band{b}.cbm'))
 predictions=[]
 for st in range(0,len(target),65536):
  sl=slice(st,min(st+65536,len(target)));x=np.c_[np.load(P/'common.npy',mmap_mode='r')[sl],np.load(P/'precise.npy',mmap_mode='r')[sl]]
  predictions.append(model.predict(x,thread_count=4).astype('float32'))
 conditional=np.concatenate(predictions);valid=target[:,b+2]>0;res=target[:,b].astype('float64')-base[:,b];tail=abs(res)>=5
 p=np.asarray(gate[:,b,0]+gate[:,b,2],dtype='float64')
 actual=p*conditional+(1-p)*mu0;oracle=tail*conditional+(~tail)*mu0
 v=valid;tv=v&tail;nt=v&~tail
 np.save(A/f'frozen_gate_delta{b}.npy',actual.astype('float32'));np.save(A/f'old_conditional_all{b}.npy',conditional)
 order=np.argsort(-p[v]);top=order[:int(np.ceil(tail[v].mean()*v.sum()))]
 def rmse(d):return float(np.sqrt(np.mean((res[v]-d[v])**2)))
 rec=dict(band=b,valid=int(v.sum()),mu_nontail_fit=mu0,C82_RMSE=float(np.sqrt(np.mean(res[v]**2))),actual_RMSE=rmse(actual),true_membership_RMSE=rmse(oracle),mean_gate_on_tail=float(p[tv].mean()),mean_gate_on_nontail=float(p[nt].mean()),precision_at_true_tail_prevalence=float(tail[v][top].mean()),tail_recall_at_gate_half=float((p[tv]>=.5).mean()),nontail_falsepositive_at_gate_half=float((p[nt]>=.5).mean()),actual_tail_RMSE=float(np.sqrt(np.mean((res[tv]-actual[tv])**2))))
 out['bands'].append(rec);print(rec,flush=True)
(A/'frozen_gate_probe.json').write_text(json.dumps(out,indent=2))
