"""Test if native two/three-scatter power improves the frozen model and/or fills the isolated power deficit."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import lightgbm as lgb
from c91_score import load,features,stats,rms,boot
R=Path(__file__).resolve().parents[1];A=R/'audit/c92_diffuse'
def prep(stage):
    rec,raw,p,br,image=load(stage);old={};new={}
    c=R/'audit/c78_clear'
    for d in c.glob('train_power_*of3'):
        ar=np.load(d/'power.npy',mmap_mode='r');old.update({j:(ar,k) for k,j in enumerate(np.load(d/'positions.npy'))})
    loc={}
    for d in sorted((R/'audit/c91_image').glob(stage+'[0-9]')):
        ls=np.load(d/'loc.npy');loc.update({e['pos']:ls[j] for j,e in enumerate(json.load(open(d/'done.json'))['rows'])})
    for d in sorted(A.glob(stage+'[0-9]')):
        ar=np.load(d/'power.npy',mmap_mode='r');new.update({v:ar[j] for j,v in enumerate(json.load(open(d/'done.json'))['positions'])})
    vw=np.stack([old[e['pos']][0][old[e['pos']][1],loc[e['pos']],14:16] for e in rec]).astype('float64')
    vn=np.stack([new[e['pos']] for e in rec]).astype('float64');rr=np.expm1(br[:,:,4].astype('float64')).clip(.25)
    coeff=rr[:,:,None]**2
    direct=10**(-br[:,:,[6,11]].astype('float64')/10);mirror=10**(-image[:,:,5:7].astype('float64')/10)
    first=direct+mirror+vw*coeff;stages=[first,first+vn[:,:,2:4]*coeff,first+(vn[:,:,2:4]+vn[:,:,4:6])*coeff]
    first_alternative=vn[:,:,:2]*coeff
    loss=lambda v:-10*np.log10(v.clip(1e-15))
    primary=features(rec,raw,p,br,image,'flat')
    single=np.concatenate([primary,loss(first),loss(first_alternative),loss(vw*coeff)],-1).astype('float32')
    multiple=np.concatenate([loss(stages[1])-loss(first),loss(stages[2])-loss(stages[1]),
                             loss(vn[:,:,2:4]*coeff),loss(vn[:,:,4:6]*coeff)],-1).astype('float32')
    pl=[20*np.log10(rr[:,:,None]*np.array([415e6,5.775e9])[None,None,:]*4*np.pi/299792458)+loss(s) for s in stages]
    return rec,raw,p,single,multiple,pl
def budget(Y,p,raw,powers,q):
    result=[]
    for b in (0,1):
        w=(raw[q,:,4+b]>.5)&(raw[q,:,6]<.01);y=raw[q,:,b];sq=(p[q,:,b]-y)**2
        g=[v[q,:,b]-y for v in powers];v=[]
        for e in g:
            under=(e>=10)&w;over=(e<=-10)&w
            v.append(dict(n=int(w.sum()),too_weak_fraction=float(under.sum()/w.sum()),too_weak_C82_SSE_fraction=float(sq[under].sum(dtype=np.float64)/sq[w].sum(dtype=np.float64)),
                          too_bright_fraction=float(over.sum()/w.sum()),model_rmse=float(np.sqrt((e[w]**2).mean())),median=float(np.median(e[w]))))
        under=(g[0]>=10)&w
        result.append(dict(orders=v,initially_10dB_too_weak_coverage=int(under.sum()),
                           power_median_gain_there=float(np.median((g[0]-g[-1])[under])),
                           remaining_10dB_too_weak_among_original=float((g[-1][under]>=10).mean())))
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);stage=ap.parse_args().stage
    rec,raw,p,core,multi,pl=prep(stage);gg=np.array([e['group'] for e in rec]);fit=gg=='fit';q=gg==('select' if stage=='screen' else 'confirm')
    choice={} if stage=='screen' else json.load(open(A/'choice.json'));scores={};pairs={}
    for mode in ['single','correct','permuted']:
        corr=multi.copy()
        if mode=='permuted':
            for j,e in enumerate(rec):
                strata=np.digitize(raw[j,:,6],[.01,.99])*48+np.digitize(pl[0][j,:,1],[80,110,145])*12+np.digitize(core[j,:,9],[4.3,5.1])*4+np.digitize(p[j,:,1]-p[j,:,0],[23,35])
                rng=np.random.default_rng([92014,e['index']]);ix=np.arange(len(strata))
                for k in np.unique(strata):
                    ii=np.flatnonzero(strata==k);ix[ii]=rng.permutation(ii)
                corr[j]=corr[j,ix]
        X=core if mode=='single' else np.concatenate([core,corr],-1)
        scores[mode]=[]
        for b in (0,1):
            mask=raw[:,:,4+b]>.5;y=raw[:,:,b];key=mode+str(b)
            if stage=='screen':
                d=lgb.Dataset(X[fit][mask[fit]],label=(y-p[:,:,b])[fit][mask[fit]])
                par=dict(objective='regression',seed=92121,num_threads=3,force_col_wise=True,deterministic=True,
                         learning_rate=.055,num_leaves=12,min_data_in_leaf=180,max_bin=63,lambda_l2=15.,verbosity=-1)
                tree=lgb.train(par,d,num_boost_round=90);sel={}
                for it in (25,55,90):
                    v=tree.predict(X[q].reshape(-1,X.shape[-1]),num_iteration=it,num_threads=3).reshape(q.sum(),-1).clip(-15,15)
                    for sc in (.3,.75,1):
                        s=stats(y[q],p[q,:,b]+sc*v,mask[q],raw[q,:,6]);sel[(it,sc)]=rms(s)[0]
                it,sc=min(sel,key=sel.get);tree.save_model(str(A/(key+'.txt')),num_iteration=it);choice[key]={'scale':sc,'trees':it}
            tree=lgb.Booster(model_file=str(A/(key+'.txt')))
            v=tree.predict(X[q].reshape(-1,X.shape[-1]),num_threads=3).reshape(q.sum(),-1).clip(-15,15)*choice[key]['scale']
            s=stats(y[q],p[q,:,b]+v,mask[q],raw[q,:,6]);bs=stats(y[q],p[q,:,b],mask[q],raw[q,:,6])
            t=np.array([e['terrain'] for e in rec])[q];ci=boot(np.stack([bs[:,0,0],s[:,0,0],s[:,0,1]],1),t)
            scores[mode].append(dict(base=rms(bs).tolist(),pred=rms(s).tolist(),gain=(rms(bs)-rms(s)).tolist(),gain95=ci))
            pairs[key]=s;pairs['C82'+str(b)]=bs
    if stage=='screen':(A/'choice.json').write_text(json.dumps(choice,indent=2)+'\n')
    be=budget(raw,p,raw,pl,q);out=dict(terrains=int(q.sum()),points=int(q.sum()*512),groups=['all','clear','shadow','partial'],
            scores=scores,physical_depth_comparison=be)
    np.savez(A/(stage+'_paired.npz'),**pairs,terrain=np.array([e['terrain'] for e in rec])[q])
    (A/(stage+'_score.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(stage,{k:[round(b['gain'][0],5) for b in v] for k,v in scores.items()},'hi-shadow physical RMSE by depth',
          [round(v['model_rmse'],3) for v in be[1]['orders']],flush=True)
if __name__=='__main__':main()
