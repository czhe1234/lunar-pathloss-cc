"""Measure the incremental held-out predictive signal of angular aliasing and native RT outputs."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from c79_select import A,read,common,sse,rms,fit_apply

def load(source):
    out={}
    pattern='angular[0-9]' if source=='angular' else 'rt_*[0-9]'
    for d in A.glob(pattern):
        if not (d/'complete.json').exists():continue
        rows=read(d/'complete.json')['records']
        if source=='angular':arr=np.load(d/'x.npy',mmap_mode='r')
        for j,r in enumerate(rows):
            a=arr[j] if source=='angular' else np.load(d/(str(r['pos'])+'.npz'))['features'].transpose(1,0,2)
            out[r['pos']]=(r,a)
    return out

def build(a,ref,b,kind,model):
    c=common(ref)
    if model=='C76':return c
    if kind=='rt':
        z=a[:,:,b].astype('float32');z[...,::2]/=20
        primary=z[:,:,:8]
        if model=='direct':v=primary
        else:
            rt=z[:,:,8:24]
            v=np.concatenate([primary,rt, z[:,:,24:32]-z[:,:,32:40],abs(z[:,:,24:32]-z[:,:,32:40])],2)
    else:
        z=a.astype('float32')
        z[:,:,:,[0,1,7]]/=6;z[:,:,:,[8,9]]/=15
        if model=='count':v=z[:,:,:,[0,6,11]]
        elif model=='shape':v=z[:,:,:,[0,1,2,5,6,7,10,11]]
        else:v=z
        v=v.reshape(*a.shape[:2],-1)
    if model=='shuffled':
        # Conditional shuffle within a map and C76 clear, predicted excess and distance/loss bins.
        d=(ref[...,3]-ref[...,2]).clip(-30,90)
        strata=np.digitize(ref[...,2],[64,72,80,90,105])*12+np.digitize(d,[19,25,33])*3+np.digitize(ref[...,6],[.01,.99])
        sh=np.empty_like(v)
        for j in range(len(ref)):
            rng=np.random.default_rng(89445+j)
            for k in np.unique(strata[j]):
                ix=np.flatnonzero(strata[j]==k);sh[j,ix]=v[j,rng.permutation(ix)]
        v=sh
    return np.concatenate([c,v,v*ref[...,6:7],v*np.clip((ref[...,3:4]-ref[...,2:3]-23)/12,0,3)],2)

def run(kind):
    data=load(kind);assert data
    rows=[data[k][0] for k in sorted(data)];a=np.stack([data[k][1] for k in sorted(data)])
    ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r')[[r['pos'] for r in rows]]
    fit=np.array([r['group']=='fit' for r in rows]);which=np.array([r['group'] for r in rows])
    models=['C76','direct','full','shuffled'] if kind=='rt' else ['C76','count','shape','full','shuffled']
    reports={};nums={}
    for group in ['select','confirm']:
        use=which==group
        if not use.any():continue
        per={}
        for model in models:
            z=[]
            for b in (0,1):
                e=ref[...,b]-ref[...,2+b]
                if model=='C76':res=e[use]
                else:
                    x=build(a,ref,b,kind,model);res,_=fit_apply(x,ref,b,fit,use);del x
                z.append(sse(res,ref[use],b))
            per[model]=np.stack(z,1)
        output={};base=rms(per['C76'])
        clusters=np.random.default_rng(7911).integers(use.sum(),size=(1500,use.sum()))
        for m,p in per.items():
            def boot(x):
                v=x[clusters].sum(1);return np.sqrt(v[...,0]/v[...,1])
            ci=np.quantile(boot(per['C76'])-boot(p),[.025,.975],axis=0)
            output[m]=dict(rmse=rms(p).tolist(),gain=(base-rms(p)).tolist(),gain95=ci.tolist())
        reports[group]=output;nums[group]=int(use.sum())
        print(kind,group,'maps',int(use.sum()),{m:(base-rms(p))[:,:2].round(5).tolist() for m,p in per.items()},flush=True)
        np.savez_compressed(A/(kind+'_'+group+'_per_map.npz'),sse=np.stack(list(per.values())),models=list(per),terrain=np.array([r['terrain'] for r,b in zip(rows,use) if b]))
    (A/(kind+'_score.json')).write_text(json.dumps(dict(fit_maps=int(fit.sum()),n=nums,zones=['all','clear','nonclear'],results=reports),indent=2)+'\n')
if __name__=='__main__':run(sys.argv[1])
