"""Measure exact discovery, absolute power and unconditional automatic replacements on fixed held-out terrain."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
def ap(a,p):
    if not a.any():return None
    ix=np.argsort(-p,kind='stable');y=a[ix];p=p[ix];j=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1]
    tp=np.cumsum(y,dtype='int64')[j];return float(np.sum(np.diff(np.r_[0,tp])*tp/(j+1))/tp[-1])
def ratio(x,y):return float(x/y) if y else None
def rmse(e):return float(np.sqrt(np.mean(e.astype('float64')**2))) if len(e) else None
def mae(e):return float(np.mean(abs(e))) if len(e) else None
def crossmap_ci(c):
    n=len(c);rng=np.random.default_rng(93321);d=c[rng.integers(n,size=(1600,n))].sum(1)
    vals=np.sqrt(d[:,0]/d[:,2])-np.sqrt(d[:,1]/d[:,2])
    return np.quantile(vals,[.025,.975]).tolist()
def winneq(pos,neg):
    if not len(pos) or not len(neg):return 0.,0
    sort=np.sort(neg);return float((np.searchsorted(sort,pos,'left')+np.searchsorted(sort,pos,'right')).sum()*.5),len(pos)*len(neg)
def match(risk,w,hit,state,dist):
    rows=[]
    for n in range(len(risk)):
        st=state[n]*16+np.clip(np.floor(dist[n]/25).astype(int),0,15)
        s=[winneq(risk[n,w[n]&hit[n]&(st==k)],risk[n,w[n]&~hit[n]&(st==k)]) for k in np.unique(st[w[n]&hit[n]])]
        rows.append(np.asarray(s).sum(0) if s else np.zeros(2))
    rows=np.array(rows);rng=np.random.default_rng(93309);boot=rows[rng.integers(len(rows),size=(1600,len(rows)))].sum(1)
    return dict(n_pairs=int(rows[:,1].sum()),auc=ratio(*rows.sum(0)),ci=np.quantile(boot[:,0]/np.maximum(boot[:,1],1),[.025,.975]).tolist())
def evaluate(pred,y,p,phy,w,event,v,space,dist,threshold):
    risk=pred[:,:,0];ordinary=pred[:,:,1];special=pred[:,:,2]
    call=risk>=threshold;tp=w&event&call;fp=w&~event&call;fn=w&event&~call
    repair=np.where(call,special,p);sq=lambda q:(y.astype('float64')-q)**2
    base2=sq(p);rep2=sq(repair)
    c=np.stack([(base2*w).sum(1),(rep2*w).sum(1),w.sum(1)],1)
    c82=float(np.sqrt(c[:,0].sum()/c[:,2].sum()));new=float(np.sqrt(c[:,1].sum()/c[:,2].sum()))
    pct=lambda n,d:ratio(int(n),int(d));stats={}
    for tag,m in [('all',w),('clear',w&(v==0)),('partial',w&(v==1)),('shadow',w&(v==2)),('local_dip',w&space)]:
        pos=m&event;k=m&call;a=pos&call
        stats[tag]=dict(n=int(m.sum()),events=int(pos.sum()),called=int(k.sum()),TP=int(a.sum()),prevalence=pct(pos.sum(),m.sum()),AP=ap(event[m],risk[m]),
            precision=pct(a.sum(),k.sum()),recall=pct(a.sum(),pos.sum()),TP_power_within1=pct((a&(abs(special-y)<=1)).sum(),a.sum()),
            recall_and_within_1dB=pct((a&(abs(special-y)<=1)).sum(),pos.sum()),
            C82_event_MAE=mae((p-y)[pos]),C82_event_RMSE=rmse((p-y)[pos]),C92_event_MAE=mae((phy-y)[pos]),
            ordinary_event_MAE=mae((ordinary-y)[pos]),ordinary_event_RMSE=rmse((ordinary-y)[pos]),
            given_event_MAE=mae((special-y)[pos]),given_event_RMSE=rmse((special-y)[pos]),
            known_event_within1=pct((pos&(abs(special-y)<=1)).sum(),pos.sum()),
            end_to_end_event_MAE=mae((repair-y)[pos]),detected_event_MAE=mae((special-y)[a]),undetected_event_MAE=mae((special-y)[pos&~call]))
    base_roi=float(base2[w].sum());g=(base2-rep2)
    res=dict(groups=stats,baseline_RMSE=c82,blind_RMSE=new,gain=c82-new,gain95=crossmap_ci(c),call_fraction=pct((w&call).sum(),w.sum()),
             false_positive=int(fp.sum()),false_negative=int(fn.sum()),fraction_of_base_SSE_saved_by_true_alerts=float(g[tp].sum()/base_roi),
             fraction_of_base_SSE_saved_by_false_alerts=float(g[fp].sum()/base_roi),
             detector_oracle_RMSE_if_only_true_alarms_repaired_exactly=float(np.sqrt(base2[w&~tp].sum()/w.sum())),
             same_terrain_state_range=match(risk,w,event,v,dist))
    return res,c
def patch_metrics(pred,y,p,ph,w,hit,state,threshold,arm):
    risk=pred[:,:,0].reshape(-1,16,7,7);state=state.reshape(risk.shape);w=w.reshape(risk.shape);hit=hit.reshape(risk.shape);cr=(risk>=threshold)&w
    y=y.reshape(risk.shape);p=p.reshape(risk.shape);ph=ph.reshape(risk.shape)
    stats={}
    for name in ('uniform','grazing'):
        tile=arm==name;out=np.zeros((len(arm),2));ex=ny=k=nchance=relax=inside_pos=0;top1=top2=0;chance=0.;joint_local=0;true_local=0
        for mode in ('x','y'):
            sa=(slice(None),slice(None),slice(None),slice(None,-1)) if mode=='x' else (slice(None),slice(None),slice(None,-1),slice(None))
            sb=(slice(None),slice(None),slice(None),slice(1,None)) if mode=='x' else (slice(None),slice(None),slice(1,None),slice(None))
            mask=w[sa]&w[sb]&(state[sa]==state[sb])&(hit[sa]!=hit[sb])&tile[:,:,None,None]
            delta=np.where(hit[sa],risk[sa]-risk[sb],risk[sb]-risk[sa]);out[:,0]+=((np.sign(delta)+1)*.5*mask).sum((1,2,3));out[:,1]+=mask.sum((1,2,3))
        for j in range(len(arm)):
            for t in np.flatnonzero(tile[j]):
                ok=w[j,t];ev=hit[j,t];vals=risk[j,t];calls=cr[j,t]
                if ev.sum()==1 and ok.sum()>=35:
                    n=np.count_nonzero(vals[ok]>vals[ev][0]);tie=np.count_nonzero(vals[ok]==vals[ev][0])
                    k+=1;top1+=(1/tie if n==0 else 0);top2+=(max(0,min(3-n,tie))/tie);chance+=1/ok.sum()
                v=ev.copy();v[[0,-1],:]=False;v[:,[0,-1]]=False
                inside_pos+=v.sum();ex+=(calls&v).sum();relax+=(maximum_filter(calls,size=3)&v).sum()
                # When all three fields share a true local shape miss. Determine neighbors only during scoring.
                for iy,ix in zip(*np.where(v)):
                    sl=np.s_[iy-1:iy+2,ix-1:ix+2];mask=ok[sl].copy();mask[1,1]=False
                    if mask.sum()<5:continue
                    def dip(a):return a[j,t,iy,ix]-np.median(a[j,t][sl][mask])
                    if dip(y)>=3:
                        true_local+=1;joint_local+=int(dip(p)<3 and dip(ph)<3)
        rng=np.random.default_rng(93731);boot=out[rng.integers(len(out),size=(1600,len(out)))].sum(1)
        q=boot[:,1]>0
        stats[name]=dict(adjacent_same_state_positive_negative_pairs=int(out[:,1].sum()),adjacent_order_AUC=ratio(*out.sum(0)),
            adjacent_auc95=np.quantile(boot[q,0]/boot[q,1],[.025,.975]).tolist(),in_patch_unique_target=int(k),top1_within_patch=ratio(top1,k),
            top3_within_patch=ratio(top2,k),random_top1=ratio(chance,k),interior_misses=int(inside_pos),exact_recall=ratio(ex,inside_pos),one_cell_neighborhood_recall=ratio(relax,inside_pos),
            severe_interior_misses_with_true_3dB_local_dip=int(true_local),both_also_miss_local_shape=int(joint_local))
    return stats
def main():
    ap0=argparse.ArgumentParser();ap0.add_argument('stage',choices=['select','confirm','patch']);a=ap0.parse_args().stage
    prefix='screen' if a=='select' else a
    target=np.load(A/(prefix+'_target.npz'));feat=np.load(A/(prefix+'_features.npz'))
    ix=target['groups']=='select' if a=='select' else np.ones(len(target['ids']),bool)
    ids=target['ids'][ix];y=target['y'][ix];w=target['valid'][ix];p=target['c82'][ix];phys=target['c92'][ix];weak=target['weak'][ix];t=target['terrain'][ix]
    assert len(set(t))==len(t) and len(ids)==len(t)
    names=feat['names'].tolist();x=feat['x'][ix];v=np.rint(2*x[:,:,names.index('state')]).astype(int);rr=np.expm1(x[:,:,names.index('range')])
    local=(target['annulus_delta'][ix]>=3) if a!='patch' else np.zeros_like(weak)
    f=json.load(open(A/'freeze.json'));report={}
    for m in f['thresholds']:
        q=np.load(A/('lookup_'+m)/(a+'.npz'));assert np.array_equal(q['ids'],ids);pred=q['pred'];res=[];cs=[]
        for b in (0,1):
            r,c=evaluate(pred[:,:,b],y[:,:,b],p[:,:,b],phys[:,:,b],w[:,:,b],weak[:,:,b],v,local[:,:,b],rr,f['thresholds'][m][b])
            if a=='patch':r['adjacent']=patch_metrics(pred[:,:,b],y[:,:,b],p[:,:,b],phys[:,:,b],w[:,:,b],weak[:,:,b],v,f['thresholds'][m][b],target['arm'])
            res.append(r);cs.append(c)
        report[m]=res
        np.savez(A/(a+'_per_terrain_'+m+'.npz'),terrain=t,sse=np.stack(cs,axis=1))
    out=dict(stage=a,selected=f['selected'],maps=len(ids),models=report)
    (A/(a+'_score.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for r in report[f['selected']]:
        val=r['groups']['all'];print(a,val['AP'],val['precision'],val['recall'],val['given_event_MAE'],r['blind_RMSE'],r['gain'])
if __name__=='__main__':main()
