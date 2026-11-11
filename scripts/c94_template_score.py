"""Same direct cell powers, different coordinate-only observation templates; compare on a common support."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c94_measurement';P=R/'audit/c93_joint_miss'
def AP(h,s):
    if not h.any():return 0.
    ix=np.argsort(-s,kind='stable');p=s[ix];y=h[ix];j=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1]
    tp=np.cumsum(y)[j];return float(np.sum(np.diff(np.r_[0,tp])*tp/(j+1))/tp[-1])
def main():
    t=np.load(P/'patch_target.npz');f=np.load(P/'patch_features.npz');ref=np.load(R/'src/c70_centroid_lookup0.npz')
    variants={'old_CC':(ref['ids'],ref['weights'])}
    for folder in sorted(A.glob('templates_*')):
        if not folder.is_dir() or not (folder/'index.json').is_file():continue
        for file in sorted(folder.glob('*.npz')):
            q=np.load(file);variants[folder.name+'/'+file.stem]=(q['ids'],q['weights'])
    assert len(variants)==37
    plan=json.load(open(P/'exact_windows.json'))['maps'];native={}
    for d in P.glob('native[0-9]'):
        for file in d.glob('*.npz'):
            if '.partial.' in file.name:continue
            z=np.load(file);native[int(z['index'])]=(z['native_ids'],z['native_inv_square'][0],z['pixel'])
    pooled={k:[] for k in variants};ys=[];base=[];phys=[];free=[];target=[];valid=[];common=[];subset=[]
    for e in plan:
        j=e['patch_map'];i=e['index'];ids,v,px=native[i];old=np.full(130050,np.nan,'float64')
        for q,val in zip(ids.ravel(),v.ravel()):
            if np.isfinite(old[q]):assert old[q]==val
            old[q]=val
        take=np.r_[tuple(np.arange(k*49,k*49+49) for k in e['windows'])]
        assert np.array_equal(t['pixels'][j,take],px)
        ok=np.ones(len(px),bool)
        for name,(ids,w) in variants.items():
            ix=ids[px];weight=w[px].astype('float64');active=weight>1e-7
            values=old[ix];availability=np.all(~active|np.isfinite(values),1);ok&=availability
            weight=np.clip(weight,0,1);weight/=weight.sum(1,keepdims=True)
            lower=np.sum(np.nan_to_num(values)*weight,axis=1)
            pooled[name].append(lower)
        clear=f['x'][j,take,f['names'].tolist().index('state')]==0
        common.append(ok&clear);ys.append(t['y'][j,take]);base.append(t['c82'][j,take]);phys.append(t['c92'][j,take])
        free.append(f['fs'][j,take]);target.append(t['weak'][j,take]);valid.append(t['valid'][j,take])
    y,p,ph,fs,hit,w=map(np.concatenate,(ys,base,phys,free,target,valid))
    m=np.concatenate(common);out={}
    for key in pooled:
        invsq=np.concatenate(pooled[key]);full=-10*np.log10(invsq.clip(1e-30))[:,None]+20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458)[None]
        aux=np.maximum(0,10**((fs-ph)/10)-1)
        directratio=10**((fs-full)/10)
        pp=fs-10*np.log10((aux+directratio).clip(1e-13));ss=[]
        for b in (0,1):
            k=m&w[:,b];risk=full[:,b]-fs[:,b]
            ss.append(dict(valid=int(k.sum()),positive=int((k&hit[:,b]).sum()),direct_alone_5dB_too_strong=int(((y[:,b]>full[:,b]+5)&k).sum()),
              AP_rank_miss=AP(hit[k,b],risk[k]),physical_direct_plus_other_RMSE=float(np.sqrt(np.mean((pp[k,b]-y[k,b])**2))),
              C92_RMSE=float(np.sqrt(np.mean((ph[k,b]-y[k,b])**2))),C82_RMSE=float(np.sqrt(np.mean((p[k,b]-y[k,b])**2)))))
        out[key]=ss
    final=dict(variants=len(variants),all_template_common_available_points=int(m.sum()),note='Same native powers and input-only templates, no label-selected production or training.',scores=out)
    (A/'templates_score.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    keys=['old_CC','templates_numpy22/numpy_mean_default','templates_numpy126/numpy_mean_default','templates_numpy22/sionna_centers_default','templates_numpy126/sionna_centers_default']
    print({k:out[k] for k in keys})
if __name__=='__main__':main()
