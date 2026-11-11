"""Fixed exemplar lookup. Unknown query powers, validity and either old predictor are never inputs."""
import argparse,json,sys,time,csv
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
A=R/'audit/c93_joint_miss'
G0=['range','vis']
G1=['range','vis','profile_obstacle','crossings','under','source_slope','graze','shot','sign_mixed','cos_spread',
    'rad_curve1','rad_s3','rad_curve8','side_curve3','horizon_in','horizon_wall']
FLOW=['first','shift','offset_delta','landed_count','landed_rays','weak_area','first_area','flow_change']
ZERO=['first','unshift','weak_area_zero','first_area']
def features(v,mode):
    names=v['names'].tolist();chs=G0 if mode=='coarse' else G1+(FLOW if mode=='flow' else ZERO if mode=='flowzero' else [])
    arr=v['rot'] if mode=='rotated' else v['x']
    x=arr[:,:,[names.index(k) for k in chs]].astype('float32')
    return x,np.rint(2*arr[:,:,names.index('state')]).astype('uint8'),v['fs'].astype('float32')
def transfer(x,t,s,train,q,qs,fs,train_fs):
    # Some validation errors are orders of magnitude larger than smooth-map variation. Use robust input scaling only.
    xv=x[train].reshape(-1,x.shape[-1]);sst=s[train].ravel();target=t['y'][train].reshape(-1,2)
    w=t['valid'][train].reshape(-1,2);hit=t['weak'][train].reshape(-1,2);exc=target-train_fs[train].reshape(-1,2)
    tids=np.repeat(np.flatnonzero(train),x.shape[1])
    qshape=q.shape[:2];q=q.reshape(-1,q.shape[-1]);qs=qs.ravel()
    med=np.median(xv,axis=0);iq=np.quantile(xv,[.15,.85],axis=0);scale=(iq[1]-iq[0]).clip(.05)
    tx=np.clip((xv-med)/scale,-5,5).astype('float32');qx=np.clip((q-med)/scale,-5,5).astype('float32')
    # Return per-band event probability, unconditional PL, and case-conditional PL.
    val=np.empty((len(q),2,3),np.float32)
    for st in (0,1,2):
        group=np.flatnonzero(qs==st)
        if not len(group):continue
        for b in (0,1):
            for cond,k in [(False,96),(True,24)]:
                eligible=(sst==st)&w[:,b]
                if cond:eligible &= hit[:,b]
                # Use all same-state FIT examples if a rare stratum has fewer than k.
                idd=np.flatnonzero(eligible)
                if not len(idd):idd=np.flatnonzero(w[:,b]&(hit[:,b] if cond else True))
                assert len(idd)>0
                rt=torch.from_numpy(tx[idd]).cuda();rs=rt.square().sum(1)
                eff=min(k,len(idd))
                for start in range(0,len(group),768):
                    use=group[start:start+768];qt=torch.from_numpy(qx[use]).cuda()
                    dist=(rs[None]-2*qt@rt.T+qt.square().sum(1,keepdim=True)).clamp_min_(0)
                    dd,ids=dist.topk(eff,largest=False)
                    ii=idd[ids.cpu().numpy()];d=dd.cpu().numpy()
                    # Each neighboring terrain contributes at most the weight of six of its exemplar locations.
                    g=tids[ii];n=np.zeros((len(use),len(train)),dtype='int32')
                    np.add.at(n,(np.arange(len(use))[:,None],g),1)
                    co=np.minimum(1.,6/np.maximum(1,np.take_along_axis(n,g,axis=1)))
                    wt=co/(1+d/(np.median(d,axis=1,keepdims=True)+.04))
                    wt=wt/wt.sum(1,keepdims=True)
                    if cond:
                        val[use,b,2]=(wt*exc[ii,b]).sum(1)
                    else:
                        val[use,b,0]=(wt*hit[ii,b]).sum(1)
                        val[use,b,1]=(wt*exc[ii,b]).sum(1)
    val=val.reshape(*qshape,2,3);val[:,:,:,1:]+=fs[:,:,:,None]
    return val
def ap(y,score):
    if not y.any():return 0.
    order=np.argsort(-score,kind='stable');a=y[order];z=score[order];n=np.r_[np.flatnonzero(z[1:]!=z[:-1]),len(z)-1]
    tp=np.cumsum(a,dtype='int64')[n];return float((np.diff(np.r_[0,tp])*tp/(n+1)).sum()/tp[-1])
def opoint(y,p):
    order=np.argsort(-p,kind='stable');k=np.r_[np.flatnonzero(p[order][1:]!=p[order][:-1]),len(p)-1]
    tp=np.cumsum(y[order],dtype='int64')[k];F=2*tp/(k+1+y.sum());idx=k[F.argmax()]
    return float(p[order[idx]])
def main():
    ap0=argparse.ArgumentParser();ap0.add_argument('--mode',required=True,choices=['coarse','angles','rotated','flow','flowzero'])
    ap0.add_argument('--stage',default='screen',choices=['screen','confirm','patch']);a=ap0.parse_args()
    gpu=gpu_guard();torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False
    sf=np.load(A/'screen_features.npz');st=np.load(A/'screen_target.npz')
    assert np.array_equal(sf['ids'],st['ids']);x,state,fs=features(sf,a.mode)
    # Only calibration supervision is available to the lookup procedure.
    groups=st['groups'];fit=groups=='fit';sel=groups=='select';names=st['terrain']
    targets={k:st[k] for k in ['valid','weak','y']};out=A/('lookup_'+a.mode);out.mkdir(exist_ok=True);ts=time.monotonic()
    if a.stage=='screen':
        rng=np.random.default_rng(9391);order=np.flatnonzero(fit);rng.shuffle(order)
        risk=np.empty((len(x),x.shape[1],2),np.float32)
        for part in np.array_split(order,5):
            q=np.zeros(len(x),bool);q[part]=True;train=fit&~q
            risk[q]=transfer(x,targets,state,train,x[q],state[q],fs[q],fs)[:,:,:,0]
        threshold=[opoint(targets['weak'][fit,:,b][targets['valid'][fit,:,b]],risk[fit,:,b][targets['valid'][fit,:,b]]) for b in (0,1)]
        pred=transfer(x,targets,state,fit,x[sel],state[sel],fs[sel],fs)
        np.savez_compressed(out/'select.npz',ids=sf['ids'][sel],pred=pred)
        ev=[]
        for b in (0,1):
            w=targets['valid'][sel,:,b];y=targets['weak'][sel,:,b][w];p=pred[:,:,b,0][w];call=p>=threshold[b];tp=(call&y).sum()
            ev.append(dict(AP=ap(y,p),precision=float(tp/max(1,call.sum())),recall=float(tp/max(1,y.sum()))))
        atomic_json(out/'select.json',dict(mode=a.mode,threshold=threshold,bands=ev,gpu=gpu,seconds=time.monotonic()-ts))
        print(a.mode,ev,flush=True);return
    assert (A/'freeze.json').exists()
    qry=np.load(A/(a.stage+'_features.npz'));qx,qs,qf=features(qry,a.mode);pred=transfer(x,targets,state,fit,qx,qs,qf,fs)
    np.savez_compressed(out/(a.stage+'.npz'),ids=qry['ids'],pred=pred)
    print(a.stage,a.mode,len(pred),'seconds',round(time.monotonic()-ts,1),flush=True)
if __name__=='__main__':main()
