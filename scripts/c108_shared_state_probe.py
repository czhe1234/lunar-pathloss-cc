"""Label-only oracle diagnostic; never an eligible predictor or competition score."""
import sys,json,time
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R))
import numpy as np
from src.c106_metrics import metrics
A=R/'audit/c108_diagnosis'
def data(part):
 p=R/'audit/c101_residual'/part;t=np.load(p/'target.npy',mmap_mode='r');base=np.load(p/'base.npy',mmap_mode='r')
 r=t[:,:2].astype(np.float64)-base.astype(np.float64);mask=t[:,2:4].astype(bool);common=mask.all(1)
 meta=json.loads((p/'done.json').read_text());pp=meta['points_per_map'];rng=np.random.default_rng({'fit':108171,'monitor':108172,'confirm':108173}[part])
 shuffle=r.copy();mean=np.zeros_like(r)
 for start in range(0,len(r),pp):
  ix=np.flatnonzero(common[start:start+pp])+start
  if len(ix):
   shuffle[ix]=r[rng.permutation(ix)];mean[start:start+pp]=r[ix].mean(0)
 weight=np.load(p/'weight.npy') if (p/'weight.npy').exists() else np.ones(len(r))
 return dict(r=r,base=np.asarray(base),target=np.asarray(t[:,:2]),mask=np.repeat(common[:,None],2,1),common=common,shuffle=shuffle,mean=mean,weight=weight)
def regress(x,y,w):
 sw=w.sum();mx=np.dot(w,x)/sw;my=np.dot(w,y)/sw
 slope=np.dot(w*(x-mx),y-my)/max(np.dot(w,(x-mx)**2),1e-20)
 return [float(my-slope*mx),float(slope)]
t=time.monotonic();fit=data('fit');coeff={}
for name in ['point','shuffle','mean']:
 v={'point':fit['r'],'shuffle':fit['shuffle'],'mean':fit['mean']}[name];c=[]
 for b in range(2):
  ok=fit['common'];c.append(regress(v[ok,1-b],fit['r'][ok,b],fit['weight'][ok]))
 coeff[name]=c
res={}
for part in ['monitor','confirm']:
 d=data(part);v={}
 v['C82_on_commonmask']=metrics(d['base'],d['base'],d['target'],d['mask'])
 for name in ['point','shuffle','mean']:
  z={'point':d['r'],'shuffle':d['shuffle'],'mean':d['mean']}[name];pred=d['base'].astype(np.float64).copy()
  for b in range(2):pred[:,b]+=coeff[name][b][0]+coeff[name][b][1]*z[:,1-b]
  v['ORACLE_otherband_'+name]=metrics(pred,d['base'],d['target'],d['mask'])
 ok=d['common'];cor=float(np.corrcoef(d['r'][ok].T)[0,1])
 res[part]=dict(metrics=v,commonmask_count=int(ok.sum()),residual_crossband_correlation=cor)
result=dict(coefficients_fitonly=coeff,results=res,scope='OFFLINE LABEL ORACLE ONLY; otherbandtrueY supplied, not available forcompetition inference; neverpromotion evidence',
 control='shuffleotherbandresidualonlyamongcommon-validpositionswithineachmap,preservemapmarginal;meancontrolmap-levelonly',
 interpretation='Alignedoraclebeatscontrols => sharedpoint-specificresidualstate;doesnotproveH/TXcannotencodeit or establishinformationfloor',
 official_VAL_used=False,seconds=time.monotonic()-t)
(A/'shared_state_ORACLE_ONLY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'coeff':coeff,'scope':result['scope'],'results':{k:{'count':z['commonmask_count'],'corr':z['residual_crossband_correlation'],'rmse':{n:m['rmse'] for n,m in z['metrics'].items()}} for k,z in res.items()},'seconds':result['seconds']}))
