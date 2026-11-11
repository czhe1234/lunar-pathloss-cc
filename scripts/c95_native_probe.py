"""Probe whether native-cell dB interpolation explains weak-site locations and power. TRAIN only."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss';D=R.parent/'data';sys.path.insert(0,str(R/'scripts'))
from c95_score import average_precision_score
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','extension','confirm']);ap.add_argument('--configs',nargs='+',required=True);a=ap.parse_args()
    plan=json.loads((A/'plan.json').read_text())
    if a.stage=='confirm':assert set(a.configs)<=set(json.loads((A/'freeze.json').read_text())['configs'])
    maps=[e for e in plan[a.stage] if all((A/a.stage/f"{e['index']}_{c}.npz").exists() for c in a.configs)]
    assert len(maps)==len(plan[a.stage]),(len(maps),len(plan[a.stage]))
    ft=np.load(P/'patch_features.npz');pt=np.load(P/'patch_target.npz');names=ft['names'].tolist()
    yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];ww=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
    all_y=[];all_valid=[];all_fs=[];all_weak=[];all_clear=[];nn={c:[] for c in a.configs};weights={c:[] for c in a.configs};available=[]
    const=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458)
    for e in maps:
        j=e['patch_map'];i=e['index'];ix=np.concatenate([np.arange(k*49,k*49+49) for k in e['windows']]);px=ft['pixels'][j,ix]
        y=np.stack([z[i].ravel()[px] for z in yy],1).astype(float);valid=np.stack([np.unpackbits(z[i])[px] for z in ww],1).astype(bool)
        fs=ft['fs'][j,ix];all_fs.append(fs);all_y.append(y);all_valid.append(valid)
        all_weak.append((y-np.maximum(pt['c82'][j,ix],pt['c92'][j,ix])>=5)&valid);all_clear.append(ft['x'][j,ix,names.index('state')]==0)
        vv=[]
        for c in a.configs:
            z=np.load(A/a.stage/f'{i}_{c}.npz');assert np.array_equal(z['pixel'],px)
            nn[c].append(z['native']);weights[c].append(z['weights']);vv.append(z['valid'])
        available.append(np.logical_and.reduce(vv))
    Y,V,FS,W,C,G=map(np.concatenate,(all_y,all_valid,all_fs,all_weak,all_clear,available))
    V &= (C&G)[:,None];r2=10**((FS[:,0]-const[0])/10)
    floor=[.0001,.001,.003,.01,.03,.1,.3,1.]
    results={}
    for c in a.configs:
        N=np.concatenate(nn[c])*r2[:,None];w=np.concatenate(weights[c])
        zero=((N==0)*w).sum(1);arithmetic=-10*np.log10((N*w).sum(1).clip(1e-100));metrics={}
        for name,score in [('linear',arithmetic),('zero_mass',zero)]:
            metrics[name]=[float(average_precision_score(W[V[:,b],b],score[V[:,b]])) for b in (0,1)]
        rr=[]
        for eta in floor:
            excess=(-10*np.log10(N+eta)*w).sum(1)
            p=FS+excess[:,None];records=[]
            for b in (0,1):
                m=V[:,b];weak=W[:,b]&m;err=p[:,b]-Y[:,b]
                records.append(dict(n=int(m.sum()),joint_weak=int(weak.sum()),AP=float(average_precision_score(W[m,b],excess[m])),
                   RMSE=float(np.sqrt((err[m]**2).mean())),weak_MAE=float(abs(err[weak]).mean()),normal_RMSE=float(np.sqrt((err[m&~W[:,b]]**2).mean()))))
            rr.append(dict(eta=eta,bands=records))
        results[c]=dict(AP_comparators=metrics,dB_domain_floor=rr)
    out=dict(stage=a.stage,maps=len(maps),scores=results,limitations='Eta is an explicitly varied, positive, isotropic non-direct power floor relative to inverse-square reference. This is a diagnostic model and not a solver-verified diffuse field. It is never used to certify a power bound. No neural training.')
    (A/f'{a.stage}_native_probe.json').write_text(json.dumps(out,indent=2)+'\n')
    for c,d in results.items():
        best=[min(d['dB_domain_floor'],key=lambda r:r['bands'][b]['RMSE']) for b in (0,1)]
        print(c,'AP',d['AP_comparators'],'best_by_band',[(z['eta'],z['bands'][b]) for b,z in enumerate(best)])
if __name__=='__main__':main()
