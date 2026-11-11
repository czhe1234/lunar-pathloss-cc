"""Precision-recall achievable within each fixed score, costs of false alerts, plus phase-specific nulls."""
import argparse,json
import numpy as np
from c93_score import A,evaluate,patch_metrics,ap
def frontier(score,positive,w,y,p,amplitude):
    s=score[w];h=positive[w];a=amplitude[w];t=y[w];b=p[w];order=np.argsort(-s,kind='stable')
    j=np.r_[np.flatnonzero(s[order][1:]!=s[order][:-1]),len(s)-1]
    tp=np.cumsum(h[order],dtype='float64')[j];pr=tp/(j+1);rc=tp/max(1,h.sum());p30=pr[rc>=.30];r30=rc[pr>=.30]
    s0=float(np.square((t-b).astype('float64')).sum());gain=np.cumsum(((t-b).astype('float64')**2-(t-a).astype('float64')**2)[order])[j]
    ix=np.r_[0,gain].argmax();ss=max(0.,gain[ix-1]) if ix else 0.;oracle=float(np.sqrt(s0/len(t))-np.sqrt((s0-ss)/len(t)))
    return dict(max_precision_at_recall_ge_30=float(p30.max()) if len(p30) else 0.,
                max_recall_at_precision_ge_30=float(r30.max()) if len(r30) else 0.,
                optimistic_one_threshold_gain_dB=oracle,optimistic_alerts=int(j[ix-1]+1) if ix else 0)
def extra(stage,tag='phase'):
    pre='screen' if stage=='select' else stage;q=np.load(A/(pre+'_target.npz'));f=np.load(A/(pre+'_features.npz'))
    use=q['groups']=='select' if stage=='select' else np.ones(len(q['ids']),bool)
    ids=q['ids'][use];w=q['valid'][use];y=q['y'][use];p=q['c82'][use];phy=q['c92'][use];weak=q['weak'][use]
    n=f['names'].tolist();x=f['x'][use];state=(x[:,:,n.index('state')]*2).round().astype(int)
    rr=np.expm1(x[:,:,n.index('range')]);local=(q['annulus_delta'][use]>=3) if stage!='patch' else np.zeros_like(weak)
    name=json.load(open(A/'freeze.json'))['selected'];base=A/('lookup_'+name)
    todo=[(name,base,json.load(open(A/'freeze.json'))['thresholds'][name])]
    if tag=='phase':
        todo+=[('phase'+str(j),A/f'phase{j}',json.load(open(A/f'phase{j}'/'choice.json'))['threshold']) for j in range(3)]
    else:todo+=[('dense',A/'dense_probe',json.load(open(A/'dense_probe/choice.json'))['threshold'])]
    report={}
    for key,path,th in todo:
        v=np.load(path/(stage+'.npz'));assert np.array_equal(v['ids'],ids);pr=v['pred'];bands=[]
        for b in (0,1):
            summ,c=evaluate(pr[:,:,b],y[:,:,b],p[:,:,b],phy[:,:,b],w[:,:,b],weak[:,:,b],state,local[:,:,b],rr,th[b])
            summ['frontier']=frontier(pr[:,:,b,0],weak[:,:,b],w[:,:,b],y[:,:,b],p[:,:,b],pr[:,:,b,2])
            if stage=='patch':summ['adjacent']=patch_metrics(pr[:,:,b],y[:,:,b],p[:,:,b],phy[:,:,b],w[:,:,b],weak[:,:,b],state,th[b],q['arm'])
            bands.append(summ)
        report[key]=bands
    if tag=='phase' and stage!='patch':
        phase=np.load(A/(pre+'_phase.npz'))['data'][use];v=np.load(A/(pre+'_geo.npz'));sh=v['sheet'][use]
        cover=(phase[:,:,0,0]>.99)&(sh[:,:,0]>.99)&(state==0)
        aux=[]
        for b in (0,1):
            mask=cover&w[:,:,b];case=weak[:,:,b];scores=phase[:,:,:,2]
            aux.append(dict(valid=int(mask.sum()),miss=int((case&mask).sum()),prevalence=float(case[mask].mean()),
                            AP_raw_direct_loss=[ap(case[mask],scores[:,:,j][mask]) for j in range(3)],
                            hitless_face_fraction_among_cases=[float(phase[:,:,j,1][mask&case].mean()) for j in range(3)]))
        report['first_receiver_complete_4B']=aux
    (A/(tag+'_'+stage+'_scores.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for k in todo:print(k[0],[(round(b['groups']['all']['AP'],3),b['frontier']) for b in report[k[0]]])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','confirm','patch']);p.add_argument('--tag',default='phase',choices=['phase','density']);a=p.parse_args();extra(a.stage,a.tag)
