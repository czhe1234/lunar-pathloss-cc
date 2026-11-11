"""Extract a same-transmitter, same-tile counterexample for auditing the original receiver transfer."""
import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver';C=R/'audit/c78_clear';D=R.parent/'data'
rows=list(csv.DictReader((D/'train_index.csv').open()));ref=np.load(C/'train_reference.npy',mmap_mode='r')
geo={};pw={};sheet={}
for k,lk in [('geom',geo),('power',pw),('sheet',sheet)]:
    for d in C.glob('train_'+k+'_*of*'):
        z=np.load(d/('geometry.npy' if k=='geom' else k+'.npy'),mmap_mode='r')
        lk.update({int(i):(z,j) for j,i in enumerate(np.load(d/'positions.npy'))})
pairs=[]
for p in A.glob('targeted*/*.npz'):
    f=np.load(p);u=int(f['pos']);row=rows[int(f['index'])];loc=f['loc'];raw=ref[u,loc];z,k=geo[u];g=z[k,loc]
    z,k=pw[u];extra=z[k,loc,6:8]+z[k,loc,14:16];z,k=sheet[u];sh=z[k,loc]
    den=4*np.pi*np.array([415e6,5.775e9])[None]/299792458
    loss=-10*np.log10(f['values'][2,:,0].clip(1e-5));shift=-10*np.log10(f['values'][3,:,0].clip(1e-5))
    margin=raw[:,:2]-20*np.log10(den*g[:,17:18]);valid=(raw[:,4:6]>.5).all(1)
    # A point where both released bands are markedly weak even though the 4B reconstructed direct field is intact.
    weak=np.flatnonzero(valid&(margin.min(1)>4)&(abs(loss)<1))
    # Opposite ordering in the very same scene, with an especially attenuated simulated direct field.
    ok=np.flatnonzero(valid&(np.abs(margin).max(1)<2)&(loss>3))
    def case(j):
        px=int(f['pixel'][j]);return dict(sample_id=row['sample_id'],terrain=row['terrain_id'],index=int(f['index']),TX=[int(row['tx_row']),int(row['tx_col'])],
            RX_row_col=list(divmod(px,256)),range_m=float(g[j,18]),min_visibility=float(g[j,:11].min()),wide_certificate=bool(g[j,15]>.5),
            actual_PL_dB=raw[j,:2].tolist(),C76_PL_dB=raw[j,2:4].tolist(),actual_loss_minus_weakest_intact_direct_dB=margin[j].tolist(),
            reconstructed_4B_direct_excess_dB=float(loss[j]),phase_shifted_direct_excess_dB=float(shift[j]),extra_power_over_FS=(extra[j]*g[j,18]**2).tolist(),
            estimated_first_sheet_hit_fraction=float(sh[j,0]),expected_hits=float(g[j,20]))
    for i in weak:
        for j in ok:
            sep=abs(g[i,18]-g[j,18])
            if sep>45:continue
            merit=min(margin[i])-abs(margin[j]).max()+min(loss[j],10)-.08*sep
            pairs.append((merit,dict(observed_weak=case(i),observed_near_direct=case(j),range_difference_m=float(sep))))
pairs.sort(key=lambda z:-z[0]);keep=[];seen=set()
for score,r in pairs:
    t=r['observed_weak']['terrain']
    if t in seen:continue
    seen.add(t);keep.append(r)
    if len(keep)==3:break
(A/'matched_counterexamples.json').write_text(json.dumps(dict(scope='Illustrations deliberately selected for contradictory ordering. Frequency estimates come only from the complete preselected panel, not from these cases.',same_terrain_pairs=keep,qualifying_pairs=len(pairs)),indent=2)+'\n')
print('pairs',len(pairs))
for r in keep:
    print(r['observed_weak']['sample_id'],r['observed_weak']['RX_row_col'],r['observed_near_direct']['RX_row_col'],r['observed_weak']['actual_loss_minus_weakest_intact_direct_dB'],r['observed_weak']['reconstructed_4B_direct_excess_dB'],r['observed_near_direct']['actual_loss_minus_weakest_intact_direct_dB'],r['observed_near_direct']['reconstructed_4B_direct_excess_dB'])
