"""Compare frequency-specific finite survival and a geometry-matched position permutation."""
import argparse,json,csv,os,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[1];A=R/'audit/c89_survival';C=R/'audit/c88_mesh';D=R.parent/'data'
def inp(groups,inds):
    ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r')
    pa=np.load(R/'audit/c78_clear/train_panel.npz');raw=ref[inds];px=pa['pixels'][inds].astype(np.int64)
    base=np.load(C/'base.npy',mmap_mode='r')
    out=np.stack([base[k].reshape(2,-1)[:,x].T for k,x in zip(inds,px)])
    info=json.load(open(C/'base_index.json'));rows=list(csv.DictReader(open(D/'train_index.csv')))
    lookup={}
    for s in sorted(A.glob('shard*')):
        a=np.load(s/'features.npy',mmap_mode='r')
        ij=json.load(open(s/'done.json'))['positions'];lookup.update({k:(a,j) for j,k in enumerate(ij)})
    allx=np.stack([lookup[k][0][lookup[k][1]] for k in inds]).astype('float32')
    terrain=np.array([rows[info['ids'][k]]['terrain_id'] for k in inds])
    return raw,out,allx,terrain
def data(x,raw,p,b,mode,inds):
    f=415e6 if b==0 else 5.775e9;freq_penalty=20*np.log10(5.775e9/415e6)
    # x[:,:,4] is a weighted log(1+distance); it contains only H and TX.
    fs=20*np.log10(np.maximum(np.expm1(x[:,:,4]),.25))+20*np.log10(4*np.pi*f/299792458)
    extra=p[:,:,b]-fs;contrast=p[:,:,1]-p[:,:,0]-freq_penalty;vis=raw[:,:,6]
    baseline=np.stack([p[:,:,b],extra,contrast,vis,x[:,:,4],x[:,:,2]],-1)
    features=[baseline,x[:,:,:16]]
    if mode!='continuum':
        feat=x[:,:,16:].copy() if mode=='placebo' else x[:,:,16:]
        if mode=='placebo':
            for n,k in enumerate(inds):
                rng=np.random.default_rng(892317+int(k))
                # Preserve state, rough number of ground crossings, range and ray density.
                strata=(np.digitize(vis[n],[.01,.99])*64+np.digitize(x[n,:,1],[.25,1.6,3.8])*16+
                        np.digitize(x[n,:,0],[1,3,5])*4+np.digitize(x[n,:,4],[4.15,4.9,5.25]))
                order=np.arange(len(strata))
                for val in np.unique(strata):
                    ii=np.flatnonzero(strata==val);order[ii]=rng.permutation(ii)
                feat[n]=feat[n,order]
        features.append(feat)
    return np.ascontiguousarray(np.concatenate(features,-1).reshape(-1,sum(t.shape[-1] for t in features)))
def stats(y,p,mask,vis):
    e=(y.astype('float64')-p)**2;g=[]
    for w in (mask,mask&(vis>.99),mask&(vis<.01),mask&(vis>=.01)&(vis<=.99)):
        g.append(np.stack([(e*w).sum(1),w.sum(1)],1))
    return np.stack(g,1)
def rms(v):
    z=v.sum(0);return np.sqrt(z[:,0]/z[:,1])
def run(args):
    b=args.band;mode=args.mode;out=A/f'{mode}_b{b}';out.mkdir(exist_ok=True)
    groups=np.array(json.load(open(C/'base_index.json'))['groups']);fit=np.flatnonzero(groups=='fit');select=np.flatnonzero(groups=='select')
    # Limit repeated terrain influence in fitting; official-style evaluation still weights valid pixels equally.
    ts=time.monotonic();raw,p,x,names=inp(groups,fit);X=data(x,raw,p,b,mode,fit);y=(raw[:,:,b]-p[:,:,b]).ravel()
    w=(raw[:,:,4+b]>.5).ravel();counts=np.unique(names,return_counts=True);weights={n:min(1.,np.sqrt(8/k)) for n,k in zip(*counts)}
    mass=np.array([weights[n] for n in names],np.float32).repeat(2048)
    tr=lgb.Dataset(X[w],label=y[w],weight=mass[w],free_raw_data=True)
    par=dict(objective='regression',learning_rate=.065,num_leaves=31,min_data_in_leaf=430,lambda_l2=20,max_bin=95,
             feature_fraction=1.,bagging_fraction=1.,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True,seed=8912)
    bst=lgb.train(par,tr,num_boost_round=120);del X,x,y,w,mass,tr
    raw,p,x,names=inp(groups,select);X=data(x,raw,p,b,mode,select);sel={}
    mask=raw[:,:,4+b]>.5;vis=raw[:,:,6];Y=raw[:,:,b]
    for tree in (25,65,120):
        v=bst.predict(X,num_iteration=tree,num_threads=4).reshape(len(p),2048).clip(-15,15)
        for scale in (.4,1.):
            s=stats(Y,p[:,:,b]+scale*v,mask,vis);sel[(tree,scale)]=rms(s)
    tree,scale=min(sel,key=lambda k:sel[k][0])
    choice=dict(band=b,mode=mode,trees=tree,scale=scale,selection={str(k):v.tolist() for k,v in sel.items()},select_C82=rms(stats(Y,p[:,:,b],mask,vis)).tolist())
    (out/'choice.json').write_text(json.dumps(choice,indent=2)+'\n');bst.save_model(str(out/'model.txt'),num_iteration=tree)
    conf=np.flatnonzero(groups=='confirm');pile=[];base=[];terr=[];regions=[];contrast=[]
    for st in range(0,len(conf),128):
        inds=conf[st:st+128];raw,p,x,names=inp(groups,inds);X=data(x,raw,p,b,mode,inds)
        v=bst.predict(X,num_iteration=tree,num_threads=4).reshape(len(p),2048).clip(-15,15)*scale
        y=raw[:,:,b];m=raw[:,:,4+b]>.5;vis=raw[:,:,6]
        pile.append(stats(y,p[:,:,b]+v,m,vis));base.append(stats(y,p[:,:,b],m,vis));terr.extend(names)
        if mode=='real':
            # Retain corrections to score the combined band's difference on joint masks.
            contrast.append(v.astype('float16'))
            if b==1:
                # Fractional native transmission probability and finite support in shadow.
                joint=(raw[:,:,4]>.5)&m&(vis<.01);loq=x[:,:,17];hiq=x[:,:,29]
                regions.append(np.array([(joint).sum(),(loq*joint).sum(),(hiq*joint).sum(),
                                  ((hiq<.05)*joint).sum(),((loq-hiq>.5)*joint).sum()]))
    v=np.concatenate(pile);bl=np.concatenate(base);old=rms(bl);new=rms(v);gain=old-new
    np.savez(out/'paired.npz',baseline=bl,candidate=v,terrain=terr,indices=conf)
    if contrast:np.save(out/'correction.npy',np.concatenate(contrast))
    result=dict(choice=choice,groups=['all','clear','shadow','partial'],maps=len(conf),terrains=len(set(terr)),base=old.tolist(),candidate=new.tolist(),
            gain=gain.tolist(),seconds=time.monotonic()-ts)
    if regions:result['shadow_joint_branch_summary']=np.sum(regions,0).tolist()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(mode,b,'chosen',tree,scale,'C82',old.round(4),'gain',gain.round(4),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['real','placebo','continuum'],required=True)
    ap.add_argument('--band',type=int,required=True);run(ap.parse_args())
