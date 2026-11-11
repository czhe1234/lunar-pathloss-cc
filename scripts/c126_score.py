"""Independent float64 RMSE against the published TRAIN maps and packed validity masks."""
from pathlib import Path
import numpy as np,json,csv,hashlib
R=Path(__file__).resolve().parents[1];A=R/'audit/c126_statefield';D=R.parent/'data'
rows=list(csv.DictReader(open(D/'train_index.csv')))
ids=json.load(open(R/'runs/c126_risk/indices.json'));assert ids==json.load(open(R/'runs/c126_state/indices.json'))
assert ids==json.load(open(R/'audit/c105_context/protocol.json'))['monitor']
terrain=np.array([int(rows[i]['terrain_id']) for i in ids]);unique,group=np.unique(terrain,return_inverse=True);n=len(unique)
br=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r');bi={i:j for j,i in enumerate(json.load(open(R/'audit/c88_mesh/base_index.json'))['ids'])}
pred={m:np.load(R/f'runs/c126_{m}/monitor_prediction.npy',mmap_mode='r') for m in ['risk','state']}
configs={m:json.load(open(R/f'runs/c126_{m}/config.json')) for m in pred}
assert configs['risk']['initial_sha256']==configs['state']['initial_sha256']
assert configs['risk']['order_sha256']==configs['state']['order_sha256']
sse={m:np.zeros((n,2)) for m in ['C82','risk','state']};count=np.zeros((n,2),np.int64)
tails={m:np.zeros(2) for m in sse};tailcount=np.zeros(2,np.int64)
for b,band in enumerate(['415','58']):
 y=np.load(D/f'train_pl{band}.npy',mmap_mode='r');mask=np.load(D/f'train_mask{band}.npy',mmap_mode='r')
 for j,i in enumerate(ids):
  valid=np.unpackbits(mask[i]).reshape(256,256).astype(bool);truth=y[i].astype('float64');base=br[bi[i],b].astype('float64');tail=valid&(np.abs(truth-base)>=5)
  count[group[j],b]+=valid.sum();tailcount[b]+=tail.sum()
  for mode in sse:
   v=base if mode=='C82' else pred[mode][j,b].astype('float64')
   assert np.isfinite(v).all();err=(v-truth)**2;sse[mode][group[j],b]+=err[valid].sum();tails[mode][b]+=err[tail].sum()
rng=np.random.default_rng(126091);boot=rng.integers(n,size=(4000,n));result={'scope':'full native valid pixels on same 128 TRAIN held-out terrains /628 maps; not official VAL','valid_pixels':count.sum(0).tolist(),'tail_pixels':tailcount.tolist(),'models':{}}
base_rmse=np.sqrt(sse['C82'].sum(0)/count.sum(0))
for mode in sse:
 score=np.sqrt(sse[mode].sum(0)/count.sum(0));result['models'][mode]=dict(RMSE=score.tolist(),tail_RMSE=np.sqrt(tails[mode]/tailcount).tolist())
 if mode!='C82':
  gain=np.sqrt(sse['C82'][boot].sum(1)/count[boot].sum(1))-np.sqrt(sse[mode][boot].sum(1)/count[boot].sum(1))
  result['models'][mode]['gain_vs_C82']=(base_rmse-score).tolist();result['models'][mode]['gain_CI95']=np.quantile(gain,[.025,.975],axis=0).T.tolist()
  saved=json.load(open(R/f'runs/c126_{mode}/result.json'));assert np.max(np.abs(score-saved['monitor']['rmse']))<1e-5
  result['models'][mode]['best_epoch']=saved['best_epoch'];result['models'][mode]['last_epoch']=saved['last_epoch']
diff=np.sqrt(sse['risk'][boot].sum(1)/count[boot].sum(1))-np.sqrt(sse['state'][boot].sum(1)/count[boot].sum(1))
result['state_gain_vs_risk']=(np.array(result['models']['risk']['RMSE'])-result['models']['state']['RMSE']).tolist();result['state_vs_risk_CI95']=np.quantile(diff,[.025,.975],axis=0).T.tolist()
result['monitor_gate_passed']=all(x>=.20 for x in result['models']['state']['gain_vs_C82']);result['official_val_used']=False
(A/'score.json').write_text(json.dumps(result,indent=2));np.savez(A/'terrain_sse.npz',count=count,**sse);print(json.dumps(result,indent=2))
