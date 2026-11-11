"""Export the two explicitly labeled cohorts, the frozen predictions, and confidence in the physics repeatability."""
import csv,json,hashlib
import numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss';D=R.parent/'data'
def main():
    t=np.load(A/'confirm_target.npz');f=np.load(A/'confirm_geo.npz');st=f['state'];dist=f['geo'][:,:,17].clip(.001)
    threshold=20*np.log10(dist[:,:,None]*np.array([415e6,5.775e9])[None,None,:]*4*np.pi/299792458)+3
    ids=t['ids'];px=t['pixels'];names=t['terrain'];y=t['y'];p=t['c82'];ph=t['c92'];weak=t['weak'];strict=t['strict'];delta=t['annulus_delta']
    sp=np.load(A/'lookup_flow/confirm.npz');dp=np.load(A/'dense_probe/confirm.npz');npred=dp['pred']
    sr=np.load(A/'strict_confirm_pred.npz');q=sr['pred'];pr=sp['pred'];th=json.load(open(A/'dense_probe/choice.json'))['threshold'];sth=json.load(open(A/'strict_threshold.json'))['threshold']
    s=np.load(A/'seed_stable_flags.npz');stable=s['status'];cert=s['any_seed_certificate']
    assert all(np.array_equal(v['ids'],ids) for v in (f,dp,sp,sr,s))
    bounds=[]
    for b in range(2):
        all_cases=(weak|strict)[:,:,b];valid=t['valid'][:,:,b];sq=(y[:,:,b].astype('float64')-p[:,:,b].astype('float64'))**2
        bounds.append(dict(union=int(all_cases.sum()),fraction_current_SSE=float(sq[all_cases].sum()/sq[valid].sum()),
                           oracle_fix_whole_union_RMSE=float(np.sqrt(sq[valid&~all_cases].sum()/valid.sum()))))
    (A/'union_budget.json').write_text(json.dumps(bounds,indent=2)+'\n')
    mask=weak|strict;clear=t['valid']&(st[:,:,None]==0);true_weak=clear&(y>threshold);left=p>threshold;right=ph>threshold
    sub={}
    for b in (0,1):
        mm=weak[:,:,b]&clear[:,:,b]
        sub[b]={'primary_clear':int(mm.sum()),'also_absolutely_weak':int((mm&true_weak[:,:,b]).sum()),
                'neither_calls_true_weak':int((mm&true_weak[:,:,b]&~left[:,:,b]&~right[:,:,b]).sum()),
                'only_C82_calls_true_weak':int((mm&true_weak[:,:,b]&left[:,:,b]&~right[:,:,b]).sum()),
                'only_C92_calls_true_weak':int((mm&true_weak[:,:,b]&~left[:,:,b]&right[:,:,b]).sum()),
                'both_call_true_weak_but_still_overpredict_power':int((mm&true_weak[:,:,b]&left[:,:,b]&right[:,:,b]).sum())}
    (A/'clear_definition_split.json').write_text(json.dumps(sub,indent=2)+'\n')
    rows=list(csv.DictReader(open(D/'train_index.csv')))
    cols=['terrain_id','sample_id','TX_row','TX_col','RX_row','RX_col','band_MHz','visibility_state',
      'both_overpredict_received_power_by_5dB','strict_clear_weak_neither_flagged','true_pathloss_dB','C82_pathloss_dB','C92_pathloss_dB',
      'true_PL_minus_max_C82_C92_dB','true_PL_minus_neighbor_median_2_to_4m_dB','direct_weak_label_threshold_dB',
      'primary_geometry_risk','primary_geometry_alarm','estimated_PL_given_primary_miss_dB','unconditional_geometry_PL_dB','strict_geometry_risk',
      'strict_geometry_alarm','estimated_PL_given_strict_miss_dB','nonnegative_diffuse_seed_certificate_for_primary','seed_check_for_primary']
    flag=['not_primary','certified_or_stable_in_two_replays','not_replayed_and_no_certificate','changed_on_at_least_one_replay']
    dest=R/'results/C93共同漏点逐像素清单_20261103.csv';n=0
    with dest.open('w',encoding='utf-8-sig',newline='') as out:
        w=csv.writer(out);w.writerow(cols)
        for j,l,b in np.argwhere(mask):
            i=int(ids[j]);row=rows[i];rr,cc=divmod(int(px[j,l]),256);is_clear=st[j,l]==0
            val=lambda a:round(float(a),4)
            w.writerow([names[j],row['sample_id'],row['tx_row'],row['tx_col'],rr,cc,[415,5775][b],['clear','partial','shadow'][st[j,l]],int(weak[j,l,b]),int(strict[j,l,b]),
                     val(y[j,l,b]),val(p[j,l,b]),val(ph[j,l,b]),val(y[j,l,b]-max(p[j,l,b],ph[j,l,b])),
                     val(delta[j,l,b]) if np.isfinite(delta[j,l,b]) else '',val(threshold[j,l,b]) if is_clear else '',
                     val(npred[j,l,b,0]),int(npred[j,l,b,0]>=th[b]),val(npred[j,l,b,2]),val(npred[j,l,b,1]),
                     val(q[j,l,b,0]) if is_clear else '',int(q[j,l,b,0]>=sth[b]) if is_clear else '',
                     val(q[j,l,b,1]) if is_clear else '',int(cert[j,l,b]),flag[stable[j,l,b]]])
            n+=1
    assert n==weak.sum()+strict.sum()-(weak&strict).sum()
    with dest.open(encoding='utf-8-sig',newline='') as ff:
        keys=set();hits=np.zeros((2,2),int)
        for row in csv.DictReader(ff):
            k=(row['sample_id'],row['RX_row'],row['RX_col'],row['band_MHz']);assert k not in keys;keys.add(k)
            j=int(row['band_MHz']=='5775');hits[j]+=np.array([int(row[col]) for col in cols[8:10]])
    assert len(keys)==n and np.array_equal(hits[:,0],weak.sum((0,1))) and np.array_equal(hits[:,1],strict.sum((0,1)))
    out=json.load(open(A/'verification.json'));out['primary_band_rows']=int(weak.sum());out['strict_band_rows']=int(strict.sum());out['overlap_rows']=int((weak&strict).sum());out['case_rows']=n
    out['CSV_SHA256']=hashlib.sha256(dest.read_bytes()).hexdigest();out['seed_replay_complete_maps']=int(s['replayed_by_map'].sum())
    (A/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
