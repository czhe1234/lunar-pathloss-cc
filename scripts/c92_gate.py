"""Can an input-only gate actually learn when the physical field is closer than C82?"""
import json,argparse
import lightgbm as lgb
import numpy as np
from c92_score import A,prep
from c89_compare import rms,stats
from c89_summary import boot
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);stage=ap.parse_args().stage
    rec,raw,p,core,multi,powers=prep(stage);groups=np.array([r['group'] for r in rec]);fit=groups=='fit';q=groups==('select' if stage=='screen' else 'confirm')
    frozen={} if stage=='screen' else json.load(open(A/'gate_choice.json'))
    results={};permap={}
    for mode in ('one','three','three_permuted'):
        full=powers[0].copy() if mode=='one' else powers[-1].copy()
        mm=multi.copy()
        if mode.endswith('_permuted'):
            for j,e in enumerate(rec):
                st=np.digitize(raw[j,:,6],[.01,.99])*24+np.digitize(p[j,:,1],[85,105,135])*6+np.digitize(core[j,:,9],[4.3,5.1])
                rng=np.random.default_rng([92621,e['index']]);order=np.arange(len(st))
                for v in np.unique(st):
                    idx=np.flatnonzero(st==v);order[idx]=rng.permutation(idx)
                # Keep the nominal map and its exposure to C82, permute only the new transport and its local increment.
                full[j]=powers[0][j]+(powers[-1][j]-powers[0][j])[order];mm[j]=mm[j,order]
        z=(full-p).clip(-160,160);X=np.concatenate([core,z,mm] if mode!='one' else [core,z],-1)
        scores=[]
        for b in (0,1):
            mask=raw[:,:,4+b]>.5;y=raw[:,:,b];err=y-p[:,:,b];key=mode+str(b);dx=z[:,:,b]
            if stage=='screen':
                hit=mask[fit]& (abs(dx[fit])>.0001);x=dx[fit][hit];target=err[fit][hit]/x
                dat=lgb.Dataset(X[fit][hit],label=target,weight=x*x)
                par=dict(objective='regression',learning_rate=.055,num_leaves=12,min_data_in_leaf=125,
                          max_bin=63,lambda_l2=100,seed=92882,num_threads=3,force_col_wise=True,deterministic=True,verbosity=-1)
                tree=lgb.train(par,dat,num_boost_round=90);sel={}
                for n in (25,55,90):
                    a=tree.predict(X[q].reshape(-1,X.shape[-1]),num_iteration=n,num_threads=3).reshape(q.sum(),-1)
                    for sc in (.4,1.):
                        v=p[q,:,b]+dx[q]*(sc*a).clip(0,1)
                        sel[(n,sc)]=rms(stats(y[q],v,mask[q],raw[q,:,6]))[0]
                n,sc=min(sel,key=sel.get);frozen[key]=[n,sc];tree.save_model(str(A/('gate_'+key+'.txt')),num_iteration=n)
            tree=lgb.Booster(model_file=str(A/('gate_'+key+'.txt')))
            alpha=(frozen[key][1]*tree.predict(X[q].reshape(-1,X.shape[-1]),num_threads=3)).reshape(q.sum(),-1).clip(0,1)
            result=p[q,:,b]+dx[q]*alpha;s=stats(y[q],result,mask[q],raw[q,:,6]);bl=stats(y[q],p[q,:,b],mask[q],raw[q,:,6])
            t=np.array([r['terrain'] for r in rec])[q];ci=boot(np.stack([bl[:,0,0],s[:,0,0],s[:,0,1]],1),t)
            scores.append(dict(rmse=rms(s).tolist(),gain=(rms(bl)-rms(s)).tolist(),gain95=ci,mean_weight=float(alpha[mask[q]].mean())))
            permap[key]=s;permap['C82'+str(b)]=bl
        results[mode]=scores
    if stage=='screen':
        frozen['selected']=min(('one','three'),key=lambda k:sum(b['rmse'][0] for b in results[k]))
        (A/'gate_choice.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (A/('gate_'+stage+'.json')).write_text(json.dumps(dict(selected=frozen['selected'],results=results),indent=2)+'\n')
    np.savez(A/('gate_'+stage+'_paired.npz'),**permap)
    print(stage,frozen['selected'],{k:[round(b['gain'][0],5) for b in v] for k,v in results.items()})
if __name__=='__main__':main()
