"""Train-free exact-neighbor test directly targeting strict clear weak points neither old model flagged."""
import csv,json,time,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from c93_score import A,R,ap,ratio,crossmap_ci
GROUP=['range','vis','profile_obstacle','crossings','under','source_slope','graze','shot','sign_mixed','cos_spread','rad_curve1','rad_s3','rad_curve8','side_curve3','horizon_in','horizon_wall','first','shift','offset_delta','landed_count','landed_rays','weak_area','first_area','flow_change']
def take(stage):
    x=np.load(A/(stage+'_features.npz'));q=np.load(A/(stage+'_target.npz'))
    names=x['names'].tolist();raw=x['x'];return dict(x=raw[:,:,[names.index(n) for n in GROUP]],fs=x['fs'],clear=raw[:,:,names.index('state')]==0,
         valid=q['valid'],y=q['y'],strict=q['strict'],p=q['c82'],phy=q['c92'],ids=q['ids'],groups=q['groups'])
def predict(a,b,cal,query):
    # Identical case/control H/TX feature space. No labels, masks or model outputs from the query.
    xr=a['x'][cal];xq=b['x'][query];sel=a['clear'][cal];qm=b['clear'][query];qq=xq[qm];qc=b['fs'][query][qm]
    y=a['y'][cal][sel];w=a['valid'][cal][sel];h=a['strict'][cal][sel];exc=y-a['fs'][cal][sel]
    rt=xr[sel];me=np.median(rt,axis=0);lo,hi=np.quantile(rt,[.15,.85],axis=0);scale=(hi-lo).clip(.05)
    rt=np.clip((rt-me)/scale,-5,5);qq=np.clip((qq-me)/scale,-5,5)
    ids=np.broadcast_to(np.arange(len(cal.nonzero()[0]))[:,None],sel.shape)[sel]
    ans=np.zeros((len(qq),2,2),np.float32)
    for band in range(2):
        for cond,num in [(0,96),(1,24)]:
            use=np.flatnonzero(w[:,band]&(h[:,band] if cond else True));n=min(len(use),num)
            tree=cKDTree(rt[use],leafsize=64)
            for j in range(0,len(qq),1200):
                dist,ii=tree.query(qq[j:j+1200],k=n,workers=4);ii=use[ii];g=ids[ii]
                k=np.zeros((len(ii),len(cal.nonzero()[0])),np.int32);np.add.at(k,(np.arange(len(ii))[:,None],g),1)
                mass=np.minimum(1.,6/np.take_along_axis(k,g,axis=1));d=dist*dist
                mass/=1+d/(np.median(d,axis=1,keepdims=True)+.04);mass/=mass.sum(1,keepdims=True)
                if cond:ans[j:j+len(ii),band,1]=qc[j:j+len(ii),band]+(mass*exc[ii,band]).sum(1)
                else:ans[j:j+len(ii),band,0]=(mass*h[ii,band]).sum(1)
    out=np.zeros((*qm.shape,2,2),np.float32);out[qm]=ans
    return out
def threshold(score,y):
    o=np.argsort(-score,kind='stable');idx=np.r_[np.flatnonzero(np.diff(score[o])!=0),len(o)-1]
    tp=np.cumsum(y[o])[idx];return float(score[o[idx[(2*tp/(idx+1+y.sum())).argmax()]]])
def main():
    a=take('screen');b=take('confirm');fit=a['groups']=='fit';sel=a['groups']=='select';order=np.flatnonzero(fit)
    np.random.default_rng(9391).shuffle(order);oof=np.zeros((*a['strict'].shape,2),np.float32);ts=time.monotonic()
    for part in np.array_split(order,5):
        q=np.zeros(len(fit),bool);q[part]=True;oof[q]=predict(a,a,fit&~q,q)
    th=[]
    for band in (0,1):
        w=fit[:,None]&a['valid'][:,:,band]&a['clear'];th.append(threshold(oof[:,:,band,0][w],a['strict'][:,:,band][w]))
    choice=A/'strict_threshold.json';choice.write_text(json.dumps(dict(threshold=th,fit_terrains=80),indent=2)+'\n')
    for stage,z,q in [('select',a,sel),('confirm',b,np.ones(len(b['ids']),bool))]:
        pred=predict(a,z,fit,q);m=z['clear'][q];w=z['valid'][q];y=z['y'][q];base=z['p'][q];evt=z['strict'][q];result=[]
        for band in (0,1):
            mask=m&w[:,:,band];h=evt[:,:,band];s=pred[:,:,band,0];val=pred[:,:,band,1];alert=(s>=th[band])&m;tp=alert&h
            es=val[h]-y[:,:,band][h];cs=(base[:,:,band]-y[:,:,band]).astype('float64')**2
            pe=(np.where(alert,val,base[:,:,band])-y[:,:,band]).astype('float64')**2;ww=w[:,:,band]
            sr=lambda x:float(np.sqrt(x[ww].mean()))
            mapcounts=np.stack([(cs*ww).sum(1),(pe*ww).sum(1),ww.sum(1)],1)
            result.append(dict(clear_valid=int(mask.sum()),strict_events=int(h.sum()),AP=ap(h[mask],s[mask]),prevalence=ratio(h.sum(),mask.sum()),alerts=int((alert&mask).sum()),
              precision=ratio(tp.sum(),(alert&mask).sum()),recall=ratio(tp.sum(),h.sum()),power_MAE_given_event=float(np.abs(es).mean()),
              original_event_MAE=float(np.abs((base[:,:,band]-y[:,:,band])[h]).mean()),given_event_within1=ratio((abs(es)<=1).sum(),h.sum()),
              detected_within1_recall=ratio((tp&(abs(val-y[:,:,band])<=1)).sum(),h.sum()),all_valid_base=sr(cs),all_valid_automatic=sr(pe),gain95=crossmap_ci(mapcounts)))
        np.savez_compressed(A/('strict_'+stage+'_pred.npz'),ids=z['ids'][q],pred=pred)
        (A/('strict_'+stage+'_scores.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(stage,result,'seconds',round(time.monotonic()-ts),flush=True)
if __name__=='__main__':main()
