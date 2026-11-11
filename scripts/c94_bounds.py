"""Audit at native receiver cells before using a direct-versus-total power inequality on raster values."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c94_measurement';P=R/'audit/c93_joint_miss';D=R.parent/'data'
def main():
    plan=json.load(open(P/'exact_windows.json'))['maps'];t=np.load(P/'patch_target.npz');f=np.load(P/'patch_features.npz')
    rows=list(csv.DictReader(open(D/'train_index.csv')))
    data={}
    for d in sorted(P.glob('native[0-9]')):
        meta=json.load(open(d/'done.json'))
        for z in d.glob('*.npz'):
            if '.partial.' in z.name:continue
            v=np.load(z);data[int(v['index'])]={k:v[k] for k in ('native_inv_square','weights','native_ids','pixel','values')}
    assert set(data)=={x['index'] for x in plan}
    original={}
    for d in P.glob('exact[0-9]'):
        for z in d.glob('*.npz'):
            if '.partial.' in z.name:continue
            v=np.load(z);original[int(v['index'])]=v['values'][0]
    hist=np.zeros((2,3,3,5));example=[];record=[];overlap=[];sup=[];qdiff=[]
    source_records=[]
    for scene,e in enumerate(plan):
        i=e['index'];j=e['patch_map'];v=data[i];w=v['weights'];native=v['native_inv_square'][0]
        px=t['pixels'][j].reshape(16,49)[e['windows']].ravel()
        ix=np.r_[tuple(np.arange(a*49,a*49+49) for a in e['windows'])];assert np.array_equal(px,v['pixel'])
        valid=t['valid'][j,ix];Y=t['y'][j,ix].astype('float64');st=f['x'][j,ix,f['names'].tolist().index('state')];geo_clear=st==0
        yes=(w>1e-7);positive=np.all((native>0)|~yes,axis=1);direct=(native*w).sum(1)
        LP=-10*np.log10(direct.clip(1e-30));ld=np.full(native.shape,np.inf);np.log10(native,where=native>0,out=ld)
        ld=np.where(native>0,-10*ld,np.inf);dB=np.where(positive,np.sum(np.where(yes,ld,0)*w,axis=1),np.inf)
        worst=np.where(positive,np.where(yes,ld,-np.inf).max(1),np.inf)
        assert np.all(w>=-1e-6) and np.max(abs(w.sum(1)-1))<1e-6 and np.all(dB>=LP-1e-4) and np.all(worst>=dB-1e-4)
        qdiff.append(abs(v['values'][0]-original[i]).max(0))
        sup.append([int(np.sum(geo_clear&positive)),int(np.sum(geo_clear))])
        native_full=[]
        for b,band in enumerate((415e6,5.775e9)):
            constant=20*np.log10(4*np.pi*band/299792458)
            bound=np.stack([LP,dB,worst],1)+constant
            mask=valid[:,b]&geo_clear
            for mode,low in enumerate(bound.T):
                for subset,keep in enumerate((mask,mask&positive,mask&~positive)):
                    err=Y[:,b]-low;ss=hist[b,mode,subset]
                    ss+=[keep.sum(),np.sum((err>3)&keep),np.sum((err>5)&keep),np.sum((err>10)&keep),np.sum(keep&np.isfinite(err))]
            native_full.append(bound)
        native_full=np.stack(native_full,2)
        m=geo_clear&valid.all(1)&positive
        deficit=Y-native_full[:,2,:];hard=m&(deficit.min(1)>=5)
        for k in np.flatnonzero(hard):
            rr,cc=divmod(int(px[k]),256);score=float(deficit[k].mean())
            example.append(dict(terrain=e['terrain'],sample=rows[i]['sample_id'],index=i,rx_row=rr,rx_col=cc,score=score,true_PL=Y[k].tolist(),
                        direct_PL_native_donors=[[float(ld[k,a]+20*np.log10(4*np.pi*band/299792458)) if w[k,a]>1e-7 else None for a in range(3)] for band in (415e6,5.775e9)],
                        donor_ids=v['native_ids'][k].tolist(),donor_weights=w[k].tolist(),
                        bound_linear_power=native_full[k,0].tolist(),bound_mean_PL=native_full[k,1].tolist(),bound_weakest_native=native_full[k,2].tolist(),
                        gap_past_weakest=deficit[k].tolist(),state='geometrically clear in registered CC candidate'))
        for k in np.flatnonzero(geo_clear&valid.any(1)):
            record.append([i,int(px[k]),e['terrain'],rows[i]['sample_id'],*np.divmod(int(px[k]),256),int(positive[k]),*valid[k].astype(int).tolist(),
                           *Y[k].tolist(),*native_full[k,0].tolist(),*native_full[k,1].tolist(),*native_full[k,2].tolist()])
        source_records.append(dict(index=i,native_faces=int(len(np.unique(v['native_ids']))),positive_fraction=float(positive.mean())))
    out={'terrains':len(plan),'windows':sum(len(e['windows']) for e in plan),'bounds':{},'source_definitions':source_records,
      'repeat_nominal_max_abs_difference':np.array(qdiff).max(0).tolist()}
    for b in (0,1):
        bands={}
        for mode,n in enumerate(['linear_power_interpolation','dB_interpolation','weakest_donor_any_convex_interpolation']):
            bands[n]={}
            for k,subset in enumerate(['all','all_native_contributors_positive','at_least_one_zero_direct_donor']):
                v=hist[b,mode,k];bands[n][subset]=dict(valid=int(v[0]),exceed3=int(v[1]),exceed5=int(v[2]),exceed10=int(v[3]),
                          finite_bound=int(v[4]),fraction_exceed5=float(v[2]/max(1,v[0])))
        out['bounds'][str(b)]=bands
    if example:
        median=np.median([x['score'] for x in example]);chosen=min(example,key=lambda x:abs(x['score']-median))
        out['selected_counterexample']=chosen
    out['dual_band_counterexample_count']=len(example)
    out['artifacts']={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        R/'.rt_packages/sionna/rt/radio_map_solvers/mesh_radio_map.py',R/'.rt_packages/sionna/rt/utils/ray_tracing.py',
        R/'audit/generation_alignment_20261101/official_sources/competition_dataset.html']}
    out['jensen_demo']=dict(native_powers=[1.,.001],weights=[.5,.5],linear_power_PL=-10*np.log10(.5005),
                             mean_dB_PL=15.,gap=15+10*np.log10(.5005))
    A.mkdir(exist_ok=True);(A/'bounds.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    with (A/'pixel_bounds.csv').open('w',encoding='utf-8-sig',newline='') as ff:
        writer=csv.writer(ff);writer.writerow(['index','pixel','terrain','sample','row','col','all_positive','valid415','valid5775','Y415','Y5775',
            'linear415','linear5775','dB415','dB5775','worst415','worst5775']);writer.writerows(record)
    print(out['repeat_nominal_max_abs_difference'],out['bounds'],out.get('selected_counterexample'))
if __name__=='__main__':main()
