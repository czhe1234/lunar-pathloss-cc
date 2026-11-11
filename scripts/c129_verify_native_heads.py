"""Independent CPU solver and full-label readout upper-bound diagnostic."""
from pathlib import Path
import json,csv,numpy as np
from collections import Counter
R=Path(__file__).resolve().parents[1];A=R/'audit/c129_fitgap'
raw=np.load(A/'conditional_native_head_stats.npz')
G,q,yy=raw['G'],raw['q'],raw['yy'];ids=raw['map_ids'];ti=raw['terrain_index'];n=len(ids)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
cnt=Counter(rows[int(i)]['terrain_id'] for i in np.load(R/'features/c38_dense/fit_map_indices.npy'))
sel=Counter(rows[int(i)]['terrain_id'] for i in ids)
w=np.array([cnt[rows[int(i)]['terrain_id']]/sel[rows[int(i)]['terrain_id']] for i in ids])
def solve(A,b,ridge):
 count=np.maximum(A[:,-1,-1],1);mx=A[:,-1,:-1]/count[:,None];my=b[:,-1]/count
 C=A[:,:-1,:-1]/count[:,None,None]-mx[:,:,None]*mx[:,None,:]
 u=b[:,:-1]/count[:,None]-mx*my[:,None];std=np.sqrt(np.maximum(np.diagonal(C,axis1=1,axis2=2),1e-16))
 C=C/std[:,:,None]/std[:,None,:];C=(C+C.transpose(0,2,1))*.5
 if ridge:
  beta=np.linalg.solve(C+ridge*np.eye(72)[None],(u/std)[:,:,None])[:,:,0]/std
 else:
  ev,U=np.linalg.eigh(C);threshold=np.maximum(ev[:,-1:]*1e-10,1e-12)
  proj=(U.transpose(0,2,1)@(u/std)[:,:,None])[:,:,0]
  beta=(U@np.divide(proj,ev,out=np.zeros_like(proj),where=ev>threshold)[:,:,None])[:,:,0]/std
 return np.concatenate([beta,(my-(beta*mx).sum(1))[:,None]],1)
def score(theta,A,b,y2):
 se=np.einsum('nbi,nbij,nbj->nb',theta,A,theta)-2*(theta*b).sum(2)+y2
 count=A[:,:,-1,-1];assert se.min()>-1e-3
 return dict(RMSE=np.sqrt((se/w[:,None]).sum(0)/(count/w[:,None]).sum(0)).tolist(),
 weighted_RMSE=np.sqrt(se.sum(0)/count.sum(0)).tolist(),valid=(count/w[:,None]).sum(0).tolist())
actual=json.loads((A/'conditional_native_heads.json').read_text());checks={}
for name in ['global','terrain_oracle','map_oracle']:
 if name=='map_oracle':
  theta=np.stack([solve(G[0,:,b],q[0,:,b],1e-3) for b in range(2)],1)
 elif name=='global':
  theta=np.stack([solve(G[0,:,b].sum(0,keepdims=True),q[0,:,b].sum(0,keepdims=True),1e-3)[0] for b in range(2)])[None].repeat(n,0)
 else:
  g=np.zeros((ti.max()+1,2,73,73));bq=np.zeros((ti.max()+1,2,73));np.add.at(g,ti,G[0]);np.add.at(bq,ti,q[0])
  theta=np.stack([solve(g[:,b],bq[:,b],1e-3) for b in range(2)],1)[ti]
 checks[name]={}
 for fold,label in [(0,'head_fit'),(1,'head_check')]:
  z=score(theta,G[fold],q[fold],yy[fold]);checks[name][label]=z
  assert np.max(np.abs(np.array(z['RMSE'])-actual['models'][name][label]['RMSE']))<1e-6
allG,allq,ally=G.sum(0),q.sum(0),yy.sum(0)
theta=np.stack([solve(allG[:,b],allq[:,b],0) for b in range(2)],1)
bound=score(theta,allG,allq,ally)
observed=actual['models']['original']
ss=sum(np.array(observed[k]['RMSE'])**2*np.array(observed[k]['valid']) for k in ['head_fit','head_check'])
nn=sum(np.array(observed[k]['valid']) for k in ['head_fit','head_check'])
rmse=np.sqrt(ss/nn)
parent=json.loads((A/'readout_verified.json').read_text())['parentFIT']
assert np.max(np.abs(rmse-parent['RMSE']))<1e-5
result=dict(status='passed',source='independentNumPyCPUsolves of retainedFP64normalstatistics; combinedoriginalhead matches separately raw-label-scored fullFIT',
 ridge_cases=checks,original_fullFIT_RMSE=rmse.tolist(),per_map_unregularized_all_native_FIT=bound,
 bound_scope='same72frozenfeatures; permapcoefficients use every valid TRAINpixel including score points. Numerical pseudoinverse cutoff1e-10. Not a prediction system or input information lower bound.',official_val_used=False)
(A/'native_heads_verified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
