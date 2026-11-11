"""Score exact-facet reflections against same-stratum permutation, at identical held-out pixels."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from c89_compare import stats,rms
from c89_summary import boot
R=Path(__file__).resolve().parents[1];A=R/'audit/c91_image'
def load(stage):
    records=[]
    ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r');pi=np.load(R/'audit/c78_clear/train_panel.npz')['pixels']
    base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
    branch={}
    for f in sorted((R/'audit/c89_survival').glob('shard*')):
        arr=np.load(f/'features.npy',mmap_mode='r')
        branch.update({j:(arr,k) for k,j in enumerate(json.load(open(f/'done.json'))['positions'])})
    for f in sorted(A.glob(stage+'[0-9]')):
        meta=json.load(open(f/'done.json'));loc=np.load(f/'loc.npy');x=np.load(f/'x.npy',mmap_mode='r')
        for j,e in enumerate(meta['rows']):
            k=e['pos'];sel=loc[j];ar,idx=branch[k]
            records.append((e,ref[k,sel],base[k].reshape(2,-1)[:,pi[k,sel]].T,ar[idx,sel].astype('float32'),x[j].astype('float32')))
    records.sort(key=lambda a:a[0]['pos'])
    return [r[0] for r in records],*[np.stack([r[k] for r in records]) for k in range(1,5)]
def features(rec,raw,p,branch,v,mode):
    fs=20*np.log10(np.maximum(np.expm1(branch[:,:,4]),.25))+20*np.log10(4*np.pi*415e6/299792458)
    ex=p-np.stack([fs,fs+20*np.log10(5.775e9/415e6)],-1);vis=raw[:,:,6]
    core=np.concatenate([p,ex,vis[:,:,None],branch[:,:,[0,1,2,4,6,11,17,29]]],-1)
    if mode=='shared':return core
    offset=0 if mode in ('flat','flat_permuted') else 16
    x=v[:,:,offset:offset+16].copy()
    if mode.endswith('_permuted'):
        for j,e in enumerate(rec):
            strata=(np.digitize(vis[j],[.01,.99])*32+np.digitize(branch[j,:,4],[4.2,4.95])*8+
                     np.digitize(ex[j,:,1],[2,10,25])*2+(branch[j,:,1]>2))
            rng=np.random.default_rng([91612,e['index']]);ii=np.arange(len(strata))
            for k in np.unique(strata):
                idx=np.flatnonzero(strata==k);ii[idx]=rng.permutation(idx)
            x[j]=x[j,ii]
    return np.concatenate([core,x],-1)
def direct_budget(raw,p,branch,v):
    r={}
    for b in (0,1):
        w=(raw[:,:,4+b]>.5)&(raw[:,:,6]<.01);y=raw[:,:,b];err=(y-p[:,:,b])**2;fs=20*np.log10(np.expm1(branch[:,:,4]).clip(.25)*4*np.pi*[415e6,5.775e9][b]/299792458)
        rr={}
        for offset,name in [(0,'flat'),(16,'tangent')]:
            direct=10**(-branch[:,:,6+5*b].astype('float64')/10)
            reflected=10**(-v[:,:,offset+5+b].astype('float64')/10)
            excess=fs-10*np.log10(np.maximum(direct+reflected,1e-20))-y
            rr[name]={}
            for tag,cond in [('any_clear_specular_path',v[:,:,offset+14]>.001),('true_total_gt_direct_plus_specular_10dB',excess>=10)]:
                a=w&cond
                rr[name][tag]={'valid_points':int(w.sum()),'fraction':float(a.sum()/w.sum()),'C82_shadow_SSE_fraction':float(err[a].sum(dtype=np.float64)/err[w].sum(dtype=np.float64))}
        r[b]=rr
    return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);stage=ap.parse_args().stage
    rec,raw,p,branch,v=load(stage);group=np.array([e['group'] for e in rec]);fit=group=='fit';q=group==('select' if stage=='screen' else 'confirm')
    choice={} if stage=='screen' else json.load(open(A/'choice.json'));results={};pairs={}
    modes=['shared','flat','tangent','flat_permuted','tangent_permuted'] if stage=='screen' else ['shared',choice['mirror'],choice['mirror']+'_permuted']
    for mode in modes:
        X=features(rec,raw,p,branch,v,mode);scores=[]
        for b in (0,1):
            mask=raw[:,:,4+b]>.5;Y=raw[:,:,b];res=Y-p[:,:,b]
            base=stats(Y[q],p[q,:,b],mask[q],raw[q,:,6]);key=mode+str(b)
            if stage=='screen':
                dataset=lgb.Dataset(X[fit][mask[fit]],label=res[fit][mask[fit]])
                para=dict(objective='regression',learning_rate=.055,num_leaves=12,min_data_in_leaf=180,max_bin=63,
                          lambda_l2=15.,verbosity=-1,num_threads=3,force_col_wise=True,deterministic=True,seed=91120)
                tree=lgb.train(para,dataset,num_boost_round=90)
                variants={}
                for it in (25,55,90):
                    vv=tree.predict(X[q].reshape(-1,X.shape[-1]),num_iteration=it,num_threads=3).reshape(q.sum(),-1).clip(-15,15)
                    for sc in (.3,.75,1.):
                        candidate=stats(Y[q],p[q,:,b]+sc*vv,mask[q],raw[q,:,6]);variants[(it,sc)]=rms(candidate)[0]
                it,sc=min(variants,key=variants.get);tree.save_model(str(A/(key+'.txt')),num_iteration=it)
                choice[key]={'trees':it,'scale':sc}
            tree=lgb.Booster(model_file=str(A/(key+'.txt')))
            vv=tree.predict(X[q].reshape(-1,X.shape[-1]),num_threads=3).reshape(q.sum(),-1).clip(-15,15)*choice[key]['scale']
            s=stats(Y[q],p[q,:,b]+vv,mask[q],raw[q,:,6]);old=rms(base);new=rms(s)
            t=np.array([e['terrain'] for e in rec])[q]
            ci=boot(np.stack([base[:,0,0],s[:,0,0],s[:,0,1]],1),t)
            scores.append(dict(base=old.tolist(),candidate=new.tolist(),gain=(old-new).tolist(),paired_gain95=ci))
            pairs[key]=s;pairs['C82'+str(b)]=base
        results[mode]=scores
    if stage=='screen':
        choice['mirror']=min(['flat','tangent'],key=lambda k:sum(r['candidate'][0] for r in results[k]))
        (A/'choice.json').write_text(json.dumps(choice,indent=2)+'\n')
    np.savez(A/(stage+'_per_map.npz'),**pairs,terrain=np.array([e['terrain'] for e in rec])[q])
    out=dict(terrains=int(q.sum()),points=int(q.sum()*512),results=results,
             groups=['all','clear','shadow','partial'],conditional_budget=direct_budget(raw[q],p[q],branch[q],v[q]))
    (A/(stage+'_scores.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(stage,{m:[round(z['gain'][0],5) for z in v] for m,v in results.items()},flush=True)
if __name__=='__main__':main()
