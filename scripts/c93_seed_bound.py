"""All-terrain deterministic certificate for a C92 miss under any nonnegative diffuse seed."""
import json
from pathlib import Path
import numpy as np
from c91_score import load as observations
from c93_seed_pairs import load as loaded_fields
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
def main():
    t=np.load(A/'confirm_target.npz');y=t['y'].astype('float64');B=t['c82'].astype('float64')
    w=t['valid'];m=t['weak'];ids=t['ids']
    rec,raw,p,br,im=observations('confirm')
    assert np.array_equal(ids,[r['index'] for r in rec]) and np.allclose(p,B) and np.allclose(y,raw[:,:,:2])
    dist=np.expm1(br[:,:,4].astype('float64')).clip(.25);freq=[415e6,5.775e9]
    F=20*np.log10(dist[:,:,None]*np.array(freq)[None,None,:]*4*np.pi/299792458)
    direct=10**(-.1*br[:,:,[6,11]].astype('float64'));mirror=10**(-.1*im[:,:,5:7].astype('float64'))
    upper=F-10*np.log10((direct+mirror).clip(1e-15))
    # Guard the numerical margin by 0.01 dB. The direct and mirror power terms are input-only and held fixed.
    cert=(y>=B+5)&(y>=upper+5.01)&w
    assert not (cert&~m).any()
    saved=loaded_fields('confirm_seed');sel=np.array([i in saved for i in ids])
    if not sel.any():raise ValueError('No actual seed replays')
    add=np.stack([saved[i] for i in ids[sel]])
    h=(y[sel,None]>=np.maximum(add,B[sel,None])+5)&w[sel,None]
    actual=m[sel]&h.all(1)
    assert not (cert[sel]&~actual).any()
    dense=np.load(A/'dense_probe/confirm.npz')['pred'];th=json.load(open(A/'dense_probe/choice.json'))['threshold']
    known=cert.copy();known[sel]=actual;possible=m.copy();possible[sel]=actual
    report={'total_terrains':len(ids),'replayed_terrains':int(sel.sum()),'bands':[]}
    for b in range(2):
        orig=m[:,:,b];true=known[:,:,b];uncertain=possible[:,:,b]&~true
        call=(dense[:,:,b,0]>=th[b])&w[:,:,b]
        assert np.count_nonzero(true|uncertain)==np.count_nonzero(possible[:,:,b])
        new=add[:,:,:,b];source=t['c92'][sel,:,b];errs=abs(new-source[:,None])
        positions=[dict(retained_fraction=float((h[:,j,:,b]&orig[sel]).sum()/orig[sel].sum()),
                        max_change_dB=float(errs[:,j][w[sel,:,b]].max()),median_p95_p99_change_dB=np.quantile(errs[:,j][w[sel,:,b]],[.5,.95,.99]).tolist()) for j in range(2)]
        safe=cert[:,:,b]
        stats=dict(original_events=int(orig.sum()),certified_any_diffuse_seed=int(safe.sum()),
                   certified_fraction=float(safe.sum()/orig.sum()),empirically_stable_on_replayed=int(actual[:,:,b].sum()),
                   replayed_original=int(orig[sel].sum()),
                   original_still_positive_in_two_extra_seeds_on_replay=float(actual[:,:,b].sum()/orig[sel].sum()),
                   confirmed_or_certified_all_maps=int(true.sum()),uncertain_on_unreplayed=int(uncertain.sum()),
                   all_terrain_stable_fraction_bounds=[float(true.sum()/orig.sum()),float((true|uncertain).sum()/orig.sum())],
                   known_stable_detection_precision=float((call&true).sum()/call.sum()),
                   known_stable_recall=float((call&true).sum()/true.sum()),
                   known_stable_depth_MAE=float(abs(dense[:,:,b,2]-y[:,:,b])[true].mean()),
                   original_alternate_stats=positions)
        report['bands'].append(stats)
    (A/'seed_stability.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    flag=np.where(~m,0,np.where(known,1,np.where(possible,2,3))).astype('int8')
    np.savez_compressed(A/'seed_stable_flags.npz',ids=ids,pixels=t['pixels'],any_seed_certificate=cert,
       replayed_by_map=sel,status=flag,labels=['not_an_original_case','verified_stable','unreplayed_and_no_bound','changed_on_replay'])
    print(report)
if __name__=='__main__':main()
