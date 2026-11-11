"""Independent NumPyFP64 masked regression and fixed signed-tail localization."""
import numpy as np
def average_precision(score,positive):
 score=np.asarray(score);positive=np.asarray(positive,dtype=bool)
 total=int(positive.sum())
 if not total:return None
 order=np.argsort(-score,kind='stable');s=score[order];y=positive[order]
 ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
 tp=np.cumsum(y,dtype=np.int64)[ends];count=ends+1
 return float(np.sum((tp/count)*np.diff(np.r_[0,tp]))/total)
def metrics(pred,base,target,mask):
 # Flatten map/pixel leading dimensions; preserve two output bands last.
 pred=np.asarray(pred,np.float64);base=np.asarray(base,np.float64);target=np.asarray(target,np.float64);mask=np.asarray(mask,bool)
 assert pred.shape==base.shape==target.shape==mask.shape and pred.shape[-1]==2
 result={};sse=[];count=[];groups=[];bands=[]
 for b in range(2):
  w=mask[...,b].reshape(-1);p=pred[...,b].reshape(-1)[w];z=base[...,b].reshape(-1)[w];y=target[...,b].reshape(-1)[w]
  assert np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(z).all()
  delta=p-z;truth=y-z;e=p-y;positive=truth>=5;negative=truth<=-5;center=~(positive|negative)
  predicted=delta>=5;tp=int((positive&predicted).sum());fp=int((~positive&predicted).sum());fn=int((positive&~predicted).sum())
  group=[]
  for g in [center,positive,negative]:
   ee=e[g];group.append(dict(count=int(g.sum()),rmse=float(np.sqrt(np.mean(ee**2))) if len(ee) else None,mae=float(np.mean(np.abs(ee))) if len(ee) else None,bias=float(np.mean(ee)) if len(ee) else None,sse=float(np.sum(ee**2))))
  ss=float(np.sum(e**2));sse.append(ss);count.append(len(e));groups.append(group)
  band=dict(positive_count=int(positive.sum()),prevalence=float(positive.mean()),predicted_positive=int(predicted.sum()),tp=tp,fp=fp,fn=fn,
   precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
   average_precision=average_precision(delta,positive),positive_power_rmse=group[1]['rmse'],positive_power_bias=group[1]['bias'],
   falsepositive_extra_sse=float(np.sum(e[predicted&~positive]**2-truth[predicted&~positive]**2)))
  bands.append(band)
 result.update(rmse=np.sqrt(np.array(sse)/count).tolist(),sse=sse,count=count,deep_fade=bands,residual_groups=groups)
 return result
