"""Separate hard contradictions already present at first RX hit from ones needing later RX crossings."""
import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss'
plan=json.loads((A/'plan.json').read_text());f=np.load(P/'patch_features.npz');names=f['names'].tolist()
raw=np.load(A/'confirm_native_predictions.npz');rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
constant=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458)
records=[];hist=np.zeros((2,4),int);dmax=0
for e in plan['confirm']:
    i=e['index'];j=e['patch_map'];base=np.load(A/'confirm'/f'{i}_base.npz');first=np.load(A/'confirm'/f'{i}_first_base.npz')
    ix=np.concatenate([np.arange(z*49,(z+1)*49) for z in e['windows']]);m=raw['terrain']==i
    assert np.array_equal(raw['pixel'][m],base['pixel']);y=raw['y'][m];valid=raw['v'][m];clear=f['x'][j,ix,names.index('state')]==0
    bf=base['native'];ff=first['native'];delta=bf-ff;dmax=max(dmax,float(np.maximum(-delta,0).max()))
    assert np.all(delta>=-np.maximum(1e-10,bf*1e-5))
    yes=base['weights']>1e-7
    def bound(n):
        pos=np.all((n>0)|~yes,axis=1);loss=np.where(n>0,-10*np.log10(n.clip(1e-300)),np.inf)
        return np.where(pos,np.where(yes,loss,-np.inf).max(1),np.inf)
    b0,b1=bound(bf),bound(ff)
    for band in (0,1):
        use=valid[:,band]&clear
        h0=(y[:,band]-b0-constant[band]>5)&use;h1=(y[:,band]-b1-constant[band]>5)&use
        hist[band]+=[h0.sum(),h1.sum(),(h0&~h1).sum(),(h1&~h0).sum()]
        for k in np.flatnonzero(h0|h1):
            pixel=int(base['pixel'][k]);rr,cc=divmod(pixel,256)
            records.append([i,rows[i]['sample_id'],band,rr,cc,float(y[k,band]),float(b0[k]+constant[band]),float(b1[k]+constant[band]),int(h0[k]),int(h1[k])])
with (A/'hard_component_cases.csv').open('w',encoding='utf-8-sig',newline='') as out:
    wr=csv.writer(out);wr.writerow(['index','sample','band_index','row','col','label_dB','full_weakest_bound_dB','first_weakest_bound_dB','full_hard','first_hard']);wr.writerows(records)
out=dict(maps=58,counts_columns=['full_direct_hard','first_hit_hard','full_hard_not_first_hard','first_hard_not_full_hard'],counts=hist.tolist(),max_negative_full_minus_first=dmax,
 interpretation='Same fixed donor set, any convex interpolation; zero first-direct donors give infinite bound. Most reconstructed hard violations require the post-first receiver component. This is component localization, not proof all official residuals are from offset.',
 first_hit_examples=[r for r in records if r[-1]==1])
(A/'component_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
