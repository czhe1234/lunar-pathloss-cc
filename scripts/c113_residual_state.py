"""Counterfactual residual-state transfer from3knownTX; not an eligible newterrain predictor."""
import csv,json,time
from pathlib import Path
import numpy as np
R=Path.cwd();D=R.parent/'data';A=R/'audit/c113_residual_state';A.mkdir(exist_ok=True);t0=time.monotonic()
plan=json.loads((R/'audit/c108_diagnosis/protocol.json').read_text());rows=list(csv.DictReader(open(D/'train_index.csv')))
ix=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());lookup={v:j for j,v in enumerate(ix)}
base=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
hm=np.load(D/'train_hm.npy',mmap_mode='r');yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
wm=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
gY,gX=np.mgrid[:256,:256]
fittrain=np.array(plan['sets']['128']['fit']).reshape(128,3).tolist();fitquery=plan['sets']['128']['same_terrain_newTX']
testgroups=np.array(plan['unseen_residual_fit_terrain32']).reshape(32,4).tolist()
groups=[(d,q,'calibration') for d,q in zip(fittrain,fitquery)]+[(z[1:],z[0],'newterrain32') for z in testgroups]
def feature(donors,q):
 ids=donors+[q];y=np.stack([np.array(v[ids],np.float64) for v in yy],1);w=np.stack([np.unpackbits(v[ids],axis=1).reshape(4,256,256).astype(bool) for v in wm],1)
 p=np.array(base[[lookup[i] for i in ids]],np.float64);r=y-p;h=np.array(hm[int(rows[q]['hm_row'])],np.float64)
 rr=np.array([int(rows[i]['tx_row']) for i in ids]);cc=np.array([int(rows[i]['tx_col']) for i in ids]);z=h[rr,cc]+3
 directions=np.stack(np.broadcast_arrays(cc[:,None,None]-gX[None],rr[:,None,None]-gY[None],z[:,None,None]-h[None]-1),-1)
 directions/=np.sqrt((directions**2).sum(-1,keepdims=True)).clip(.25)
 dot=(directions[:3]*directions[3:4]).sum(-1).clip(-1,1)
 common=w[3]&w[:3].any(0);simple=w[:3].astype(np.float64);angular=simple*np.exp(16*(dot[:,None]-1))
 mean=(np.where(w[:3],r[:3],0)*simple).sum(0)/simple.sum(0).clip(1e-30)
 kernel=(np.where(w[:3],r[:3],0)*angular).sum(0)/angular.sum(0).clip(1e-30)
 chosen=angular.argmax(0);nearest=np.take_along_axis(r[:3],chosen[None],axis=0)[0]
 return r[3],common,{'mean':mean,'angular':kernel,'nearestangular':nearest}
# OLS with intercept and one residual-state scalar, calibrated on128 FITterrains only.
sums={m:np.zeros((2,5)) for m in ['mean','angular','nearestangular']}
for donors,q,kind in groups[:128]:
 target,mask,ff=feature(donors,q)
 for m,f in ff.items():
  for b in range(2):
   x=f[b][mask[b]];y=target[b][mask[b]]
   sums[m][b]+=np.array([len(x),x.sum(),y.sum(),(x*x).sum(),(x*y).sum()])
coef={}
for m,s in sums.items():
 c=[]
 for n,sx,sy,sxx,sxy in s:
  beta=(sxy-sx*sy/n)/max(sxx-sx*sx/n,1e-20);c.append([float(sy/n-beta*sx/n),float(beta)])
 coef[m]=c
records=[]
for donors,q,kind in groups[128:]:
 target,mask,ff=feature(donors,q);stats={};n=mask.sum((1,2));weak=mask&(target>=5);nw=weak.sum((1,2))
 for m in ['C82','mean','angular','nearestangular']:
  correction=np.zeros_like(target)
  if m!='C82':
   for b in range(2):correction[b]=coef[m][b][0]+coef[m][b][1]*ff[m][b]
  err=np.where(mask,correction-target,0)
  stats[m]=dict(sse=(err**2).sum((1,2)).tolist(),weak_sse=np.where(weak,err**2,0).sum((1,2)).tolist())
 records.append(dict(query=q,donors=donors,count=n.tolist(),weak_count=nw.tolist(),metrics=stats))
count=np.array([z['count'] for z in records]).sum(0);nw=np.array([z['weak_count'] for z in records]).sum(0)
metrics={}
for m in ['C82','mean','angular','nearestangular']:
 s=np.array([z['metrics'][m]['sse'] for z in records]).sum(0);ws=np.array([z['metrics'][m]['weak_sse'] for z in records]).sum(0)
 metrics[m]=dict(rmse=np.sqrt(s/count).tolist(),weak_power_rmse=np.sqrt(ws/nw).tolist())
result=dict(scope='LABELCONTEXTCOUNTERFACTUAL:3knownTXresidualfieldsprovidedfornewterrain32;notavailableincompetitiontest,no promotion evidence',
 coefficients128FIT=coef,metrics_newterrain32=metrics,count=count.tolist(),weak_count=nw.tolist(),perterrain=records,
 interpretation='Testsresidualcoherenceandwhetherreceiverangularlatentwouldhelpifknown;doesnotshowlatentcanbepredictedfromH/TX',
 official_VAL_used=False,seconds=time.monotonic()-t0)
(A/'ORACLE_CONTEXT_ONLY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='perterrain'}),flush=True)
