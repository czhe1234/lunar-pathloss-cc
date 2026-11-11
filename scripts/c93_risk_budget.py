"""Signed residuals in geometric risk bins. Quantify whether correcting false alerts more conservatively helps."""
import csv,json
from pathlib import Path
import numpy as np
from c93_score import A,crossmap_ci
def main():
    d=A/'dense_probe';o=np.load(d/'oof.npz');c=np.load(A/'screen_target.npz');fit=c['groups']=='fit';assert np.array_equal(o['ids'],c['ids'][fit])
    f=np.load(A/'screen_features.npz');names=f['names'].tolist();ix=names.index('state')
    cp=o['pred'][:,:,:,0];st=(f['x'][fit,:,:,][...,ix]*2).round().astype(int)
    cal_y=c['y'][fit];cal_w=c['valid'][fit];cal_p=c['c82'][fit];cal_e=cal_y-cal_p
    q=np.load(A/'confirm_target.npz');v=np.load(d/'confirm.npz');assert np.array_equal(q['ids'],v['ids'])
    x=np.load(A/'confirm_features.npz');state=(x['x'][...,ix]*2).round().astype(int)
    y=q['y'];p=q['c82'];w=q['valid'];event=q['weak'];pr=v['pred'];res=y.astype('float64')-p
    thresholds=json.load(open(d/'choice.json'))['power_threshold'];summary={}
    for b in (0,1):
        pred=p[:,:,b].copy();oracle=pred.copy();tables={}
        for z in range(3):
            a=cal_w[:,:,b]&(st==z);edges=np.quantile(cp[:,:,b][a],[.5,.8,.9,.95,.99]);cb=np.digitize(cp[:,:,b],edges);tb=np.digitize(pr[:,:,b,0],edges)
            for k in range(6):
                train=a&(cb==k);test=w[:,:,b]&(state==z)&(tb==k)
                # Take only positive signed error, without using event truth at query or any second predictor.
                value=float(cal_e[:,:,b][train].mean()) if train.any() else 0.;adj=max(0.,value)
                pred[test]+=adj
                if test.any():oracle[test]+=max(0.,float(res[:,:,b][test].mean()))
                tables[f'{z}_{k}']=dict(cal_count=int(train.sum()),mean_cal_residual=value,
                    query_count=int(test.sum()),query_mean_signed_error=float(res[:,:,b][test].mean()) if test.any() else None)
        sq=lambda v:((y[:,:,b].astype('float64')-v)**2*w[:,:,b]).sum(1)
        cc=w[:,:,b].sum(1);base=sq(p[:,:,b]);new=sq(pred);ideal=sq(oracle);eq=lambda v:float(np.sqrt(v.sum()/cc.sum()))
        hit=w[:,:,b]&(pr[:,:,b,0]>=thresholds[b]);full=p[:,:,b].copy();full[hit]=pr[:,:,b,2][hit]
        al=int(hit.sum());t=int((event[:,:,b]&hit).sum())
        summary[b]=dict(baseline=eq(base),blind_group_correction=eq(new),gain=eq(base)-eq(new),gain95=crossmap_ci(np.stack([base,new,cc],1)),
               optimal_positive_group_bias_same_query_RMSE=eq(ideal),power_threshold=thresholds[b],power_threshold_calls=al,power_threshold_true=t,
               power_threshold_blind_RMSE=eq(sq(full)),risk_bins=tables)
    (A/'risk_budget.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print([(v['gain'],v['power_threshold_calls']) for v in summary.values()])
if __name__=='__main__':main()
