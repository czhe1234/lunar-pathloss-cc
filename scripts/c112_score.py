"""Paired TX response decomposition of frozen predictions against TRAIN labels."""
import sys,json,csv,time
from pathlib import Path
import numpy as np
R=Path.cwd();sys.path.insert(0,str(R))
from src.c106_metrics import average_precision
A=R/'audit/c112_TXresponse';D=R.parent/'data';plan=json.loads((A/'protocol.json').read_text())
pred={}
for shard in [0,1]:
 p=A/f'shard{shard}';assert (p/'done.json').exists()
 ids=json.loads((p/'ids.json').read_text());v=np.load(p/'prediction.npy',mmap_mode='r')
 pred.update({i:v[j] for j,i in enumerate(ids)})
assert set(pred)==set(plan['ids'])
rows=list(csv.DictReader(open(D/'train_index.csv')));y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
mask=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
gx,gy=np.meshgrid(np.arange(256),np.arange(256));records=[];rank={c:[[],[]] for c in ['exact0','unit1m']};truth={c:[[],[]] for c in rank}
for pair in plan['pairs']:
 i,j=pair['ids'];cat=pair['category'];pos=[(int(rows[k]['tx_row']),int(rows[k]['tx_col'])) for k in [i,j]]
 remote=np.ones((256,256),bool)
 for r,c in pos:remote&=((gx-c)**2+(gy-r)**2>=100)
 bands=[]
 for b in range(2):
  w=remote&np.unpackbits(mask[b][i]).reshape(256,256).astype(bool)&np.unpackbits(mask[b][j]).reshape(256,256).astype(bool)
  a0=np.asarray(y[b][i],np.float64)[w];a1=np.asarray(y[b][j],np.float64)[w]
  p0=np.asarray(pred[i][b],np.float64)[w];p1=np.asarray(pred[j][b],np.float64)[w]
  dy=a1-a0;dp=p1-p0;e0=p0-a0;e1=p1-a1;total=(e0**2+e1**2)/2;shared=((e0+e1)/2)**2;diff=((e0-e1)/2)**2
  assert np.allclose(total,shared+diff,rtol=1e-10,atol=1e-10)
  positive=abs(dy)>5;flag=abs(dp)>5;bad=positive&(abs(dp-dy)>5)
  rank[cat][b].append(abs(dp).astype(np.float32));truth[cat][b].append(positive)
  bands.append(dict(count=len(dy),label_delta_SSE=float((dy**2).sum()),pred_delta_SSE=float((dp**2).sum()),delta_error_SSE=float(((dp-dy)**2).sum()),
   permap_error_SSE=float(total.sum()),shared_error_SSE=float(shared.sum()),differential_error_SSE=float(diff.sum()),
   permap_error_SSE_on_actual_changes_gt5=float(total[positive].sum()),permap_error_SSE_on_bad_responses=float(total[bad].sum()),
   actual_changes_gt5=int(positive.sum()),tp=int((positive&flag).sum()),fp=int((~positive&flag).sum()),fn=int((positive&~flag).sum()),
   correct_sign_true_positive=int(((np.sign(dy)==np.sign(dp))&positive&flag).sum())))
 records.append(dict(terrain=pair['terrain'],ids=[i,j],category=cat,bands=bands))
summary={}
for cat in ['exact0','unit1m']:
 sub=[z for z in records if z['category']==cat];bands=[]
 for b in range(2):
  total={k:sum(v['bands'][b][k] for v in sub) for k in sub[0]['bands'][b]}
  ss=total['permap_error_SSE'];n=total['count'];tp=total['tp'];fp=total['fp'];fn=total['fn']
  bands.append(dict(count=n,label_delta_RMSE=float(np.sqrt(total['label_delta_SSE']/n)),prediction_delta_RMSE=float(np.sqrt(total['pred_delta_SSE']/n)),
   response_error_RMSE=float(np.sqrt(total['delta_error_SSE']/n)),response_energy_explained=1-total['delta_error_SSE']/max(total['label_delta_SSE'],1e-20),
   base_permap_RMSE_on_commonpairs=float(np.sqrt(ss/n)),error_shared_between_TX_fraction=total['shared_error_SSE']/ss,
   error_differential_between_TX_fraction=total['differential_error_SSE']/ss,
   fraction_baseSSE_on_actual_changes_gt5=total['permap_error_SSE_on_actual_changes_gt5']/ss,
   fraction_baseSSE_on_bad_responses=total['permap_error_SSE_on_bad_responses']/ss,
   large_change_count=total['actual_changes_gt5'],precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
   sign_correct_among_TP=total['correct_sign_true_positive']/tp if tp else None,
   AP=average_precision(np.concatenate(rank[cat][b]),np.concatenate(truth[cat][b]))))
 summary[cat]=dict(pairs=len(sub),terrains=len(set(z['terrain'] for z in sub)),bands=bands)
result=dict(summary=summary,perpair=records,scope='TRAIN-side frozenC82development response analysis,notnewmodelvalidation',
 definition='delta=PL_TX2-PL_TX1;receiver>=10mfrombothTX,commonvalidmask. Meanpaired error=shared^2+differential^2 exactly.',
 no_new_training=True,official_VAL_used=False)
(A/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary),flush=True)
