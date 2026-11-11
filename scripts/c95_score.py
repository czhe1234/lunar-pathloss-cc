"""Score frozen scene controls. Read only TRAIN targets for the requested terrain stage."""
import argparse,csv,json
from pathlib import Path
import numpy as np
def average_precision_score(y,score):
    ix=np.argsort(-score,kind="stable");yy=np.asarray(y,dtype=bool)[ix];ss=score[ix]
    end=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1];tp=np.cumsum(yy)[end]
    return np.sum(np.diff(np.r_[0,tp])*tp/(end+1))/max(1,yy.sum())
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss';D=R.parent/'data'
CONSTANT=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458)
def stat(y,p,m):
    if not m.any():return dict(n=0)
    err=(p-y)[m]
    return dict(n=int(m.sum()),RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),bias=float(np.mean(err)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','extension','confirm']);ap.add_argument('--partial',action='store_true');ap.add_argument('--configs',nargs='*');a=ap.parse_args()
    plan=json.loads((A/'plan.json').read_text());configs=a.configs or list(plan['configs'])
    if a.stage=='confirm':assert set(configs)<=set(json.loads((A/'freeze.json').read_text())['configs'])
    entries=[e for e in plan[a.stage] if all((A/a.stage/f"{e['index']}_{c}.npz").exists() for c in configs)]
    if not a.partial:assert len(entries)==len(plan[a.stage]),(len(entries),len(plan[a.stage]))
    if not entries:print('none complete');return
    rows=list(csv.DictReader(open(D/'train_index.csv')));yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];ww=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
    ft=np.load(P/'patch_features.npz');pt=np.load(P/'patch_target.npz')
    ns=ft['names'].tolist();where={int(i):j for j,i in enumerate(ft['ids'])}
    Y=[];V=[];FS=[];CLEAR=[];WEAK=[];OLD=[];BASE=[];values={c:[] for c in configs};native={c:[] for c in configs};good={c:[] for c in configs};terrain=[]
    for e in entries:
        i=e['index'];j=where[i];ix=np.concatenate([np.arange(w*49,(w+1)*49) for w in e['windows']]);px=ft['pixels'][j,ix]
        y=np.stack([v[i].ravel()[px] for v in yy],1).astype('float64');v=np.stack([np.unpackbits(z[i])[px] for z in ww],1).astype(bool)
        old=pt['c92'][j,ix].astype('float64');base=pt['c82'][j,ix].astype('float64')
        Y.append(y);V.append(v);FS.append(ft['fs'][j,ix]);CLEAR.append(ft['x'][j,ix,ns.index('state')]==0)
        WEAK.append((y-np.maximum(old,base)>=5)&v);OLD.append(old);BASE.append(base);terrain.extend([i]*len(px))
        for c in configs:
            z=np.load(A/a.stage/f'{i}_{c}.npz');assert np.array_equal(px,z['pixel'])
            values[c].append(z['average']);n=z['native'];w=z['weights'];yes=w>1e-7;positive=np.all((n>0)|~yes,1)
            ld=np.where(n>0,-10*np.log10(n.clip(1e-300)),np.inf)
            worst=np.where(positive,np.where(yes,ld,-np.inf).max(1),np.inf)
            native[c].append(np.stack([worst,positive],1));good[c].append(z['valid'])
    y,v,fs,cl,weak,old,base=map(np.concatenate,(Y,V,FS,CLEAR,WEAK,OLD,BASE))
    direct={c:np.concatenate(values[c]) for c in configs};nb={c:np.concatenate(native[c]) for c in configs}
    common=np.logical_and.reduce([np.concatenate(good[c]) for c in configs]);v=v&common[:,None]
    assert 'base' in configs
    olddir=direct['base'][:,None]*10**(-CONSTANT[None,:]/10)
    other=np.maximum(10**(-old/10)-olddir,0)
    records={}
    for c in configs:
        direct_pl=CONSTANT[None,:]-10*np.log10(direct[c].clip(1e-300))[:,None]
        # Counterfactual control only: hold C92 non-direct term fixed. Not exact full propagation for a changed scene.
        total=other+direct[c][:,None]*10**(-CONSTANT[None,:]/10)
        defined=total>0
        p=np.where(defined,-10*np.log10(total.clip(1e-300)),old)
        score=direct_pl-fs
        rr=[]
        for b in (0,1):
            m=v[:,b]&cl;w=weak[:,b]&m;neg=m&~weak[:,b];hard=(y[:,b]-direct_pl[:,b])>5
            apv=float(average_precision_score(weak[m,b],score[m,b])) if w.any() else None
            alarms=m&(score[:,b]>3);tp=alarms&weak[:,b]
            worst=nb[c][:,0]+CONSTANT[b]
            rr.append(dict(valid_clear=int(m.sum()),joint_weak=int(w.sum()),AP=apv,
                predicted_weak_attenuation3=int(alarms.sum()),true_positive=int(tp.sum()),
                precision=float(tp.sum()/max(1,alarms.sum())),recall=float(tp.sum()/max(1,w.sum())),
                direct_interpolated_5dB_too_strong=int((hard&m).sum()),
                all_native_positive=int(((nb[c][:,1]>.5)&m).sum()),
                weakest_native_5dB_too_strong=int(((y[:,b]-worst>5)&m).sum()),
                undefined_total_cells=int((~defined[:,b]&v[:,b]).sum()),
                undefined_weak_cells=int((~defined[:,b]&w).sum()),
                counterfactual_all=stat(y[:,b],p[:,b],v[:,b]),
                counterfactual_clear=stat(y[:,b],p[:,b],m),
                counterfactual_joint_weak=stat(y[:,b],p[:,b],w),
                counterfactual_clear_normal=stat(y[:,b],p[:,b],neg),
                original_C92_clear=stat(y[:,b],old[:,b],m),
                original_C82_clear=stat(y[:,b],base[:,b],m)))
        records[c]=rr
    out=dict(stage=a.stage,maps=len(entries),indices=[e['index'] for e in entries],windows=sum(len(e['windows']) for e in entries),common_supported=int(common.sum()),scores=records,
       caution='Direct replay exact for current candidate and nominal lattice. Counterfactual total changes direct only and holds clipped C92 non-direct power fixed. If it gives zero total, retain original C92 and separately count undefined cells. No arbitrary epsilon loss is scored. Not a full new propagation solution, official score, or fitted model.')
    (A/f'{a.stage}_score.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('stage',a.stage,'maps',len(entries),'points',len(y))
    for c,bs in records.items():print(c,[[round(r['AP'],4) if r['AP'] is not None else None,r['direct_interpolated_5dB_too_strong'],round(r['counterfactual_joint_weak'].get('MAE',0),3),round(r['counterfactual_clear'].get('RMSE',0),3)] for r in bs])
if __name__=='__main__':main()
