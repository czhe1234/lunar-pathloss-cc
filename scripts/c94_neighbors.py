"""Whether a permissive one-metre neighboring donor can remove a native-power bound contradiction."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c94_measurement';P=R/'audit/c93_joint_miss'
def main():
    plan=json.load(open(P/'exact_windows.json'))['maps'];t=np.load(P/'patch_target.npz');f=np.load(P/'patch_features.npz')
    column=f['names'].tolist().index('state');source={}
    for d in P.glob('native[0-9]'):
        for p in d.glob('*.npz'):
            if '.partial.' in p.name:continue
            z=np.load(p);source[int(z['index'])]=(z['native_inv_square'][0],z['pixel'],z['weights'],z['values'][0,:,0])
    counts=np.zeros((2,3,4),int);controls=np.zeros((2,2,3),int)
    for e in plan:
        i=e['index'];j=e['patch_map'];raw,px,w,meanratio=source[i];rr,cc=np.divmod(px,256)
        take=np.r_[tuple(np.arange(a*49,a*49+49) for a in e['windows'])]
        y=t['y'][j,take];valid=t['valid'][j,take];clear=f['x'][j,take,column]==0
        active=w>1e-7;positive=np.all((raw>0)|~active,1)
        for k in np.flatnonzero(clear&positive):
            for b,freq in enumerate([415e6,5.775e9]):
                if not valid[k,b]:continue
                constant=20*np.log10(4*np.pi*freq/299792458);worst=(constant-10*np.log10(raw[k][active[k]])).max()
                hard=y[k,b]>worst+5
                for r in (0,1,2):
                    near=(abs(rr-rr[k])<=r)&(abs(cc-cc[k])<=r)
                    if near.sum()!=(2*r+1)**2:continue
                    powers=raw[near][active[near]];zero=(powers<=0).any()
                    candidate=np.inf if zero else (constant-10*np.log10(powers)).max()
                    if hard:counts[b,r]+=[1,int(y[k,b]<=candidate),int(y[k,b]>candidate+5),int(zero)]
                    if r==1:
                        group=0 if hard else 1
                        controls[b,group]+=[1,int(zero),int(meanratio[k]<.8)]
    out=dict(columns=['counterexamples_with_complete_neighborhood','neighbor_direct_weak_enough_to_allow_label',
                     'still_at_least_5dB_beyond_all_neighbor_donors','neighborhood_contains_zero_direct'],
             band_radius_counts=counts.tolist(),one_meter_comparator_columns=['positions','contains_zero_direct','central_mean_direct_under_minus1dB'],
             one_meter_groups=['hard_max_donor_counterexample','remaining_all_positive_native_positions'],
             one_meter_controls=controls.tolist(),
             limitation='A label-assisted permissive bound, not a learned map or a blind prediction. Many other cells also have zero-direct neighbors.')
    (A/'neighbor_bound.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
