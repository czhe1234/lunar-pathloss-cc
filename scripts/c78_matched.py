"""Matched positional contrasts and an explicitly label-assisted weak-spot diagnostic."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear';sys.path.insert(0,str(R/'scripts'))
from c78_geometry_attribution import read_parts
def run(split):
    g,_=read_parts(split,'geom');p,_=read_parts(split,'power');ref=np.load(A/f'{split}_reference.npy',mmap_mode='r');pan=np.load(A/f'{split}_panel.npz')
    cal=pan['calibration'] if split=='train' else np.zeros(len(ref),bool)
    sheet=np.empty((*ref.shape[:2],10),np.float32)
    for d in sorted(A.glob(f'{split}_sheet_*of*')):sheet[np.load(d/'positions.npy')]=np.load(d/'sheet.npy',mmap_mode='r')
    allout={};oracle_weights={};strength_thresholds={}
    for b,freq in enumerate((415e6,5.775e9)):
        clear=(ref[...,6]>.99)&(ref[...,4+b]>.5);y=ref[...,b];pred=ref[...,2+b];e=y-pred
        k=20*np.log10(4*np.pi*freq/299792458);d=g[...,18].clip(.001);fs=k+20*np.log10(d);upper=k+20*np.log10(g[...,17].clip(.001))
        actual=y>upper+3;flag=pred>upper+3;band_report={}
        # The calibration table uses only the old terrain partition. The second table sees the outcome flag and is NOT deployable.
        level=np.digitize(pred-fs,[-2,0,2,5]);oracle_key=2*level+actual
        if split=='train':
            use=cal[:,None]&clear;off=[]
            for ind,num in [(level,5),(oracle_key,10)]:
                cnt=np.bincount(ind[use],minlength=num);val=np.bincount(ind[use],weights=e[use],minlength=num)/(cnt+20)
                off.append(val)
            oracle_weights[str(b)]=[x.tolist() for x in off]
        else:off=[np.array(x) for x in json.load(open(A/'weak_marker_calibration.json'))[str(b)]]
        for zone,sel in [('clear',clear),('stable',clear&(g[...,:11].min(2)>=.99)),('wide',clear&(g[...,15]>.5))]:
            mask=sel&(~cal[:,None] if split=='train' else True);weak=mask&actual;n=mask.sum()
            cells={}
            for name,grp in [('pred_normal_true_normal',~flag&~actual),('pred_weak_true_normal',flag&~actual),('pred_normal_true_weak',~flag&actual),('pred_weak_true_weak',flag&actual)]:
                use=mask&grp;err=e[use].astype('float64');cells[name]=dict(n=int(use.sum()),share=use.sum()/n,sse_share=(err@err)/np.sum(e[mask].astype('float64')**2),bias=float(err.mean()) if len(err) else None)
            rms=lambda v:float(np.sqrt(np.mean(v.astype('float64')**2)))
            paired=mask&(ref[...,4+(1-b)]>.5);other_k=20*np.log10(4*np.pi*[415e6,5.775e9][1-b]/299792458);other_weak=ref[...,1-b]>other_k+20*np.log10(g[...,17].clip(.001))+3
            band_report[zone]=dict(count=int(n),actual_weak_share=weak.sum()/n,recall=(weak&flag).sum()/weak.sum(),precision=(weak&flag).sum()/max((mask&flag).sum(),1),cells=cells,baseline_RMSE=rms(e[mask]),input_marker_RMSE=rms((e-off[0][level])[mask]),
                label_assisted_RMSE=rms((e-off[1][oracle_key])[mask]),other_band_also_weak_given_weak=(paired&actual&other_weak).sum()/(paired&actual).sum(),
                valid_145_in_deficit=int((weak&(y==145)).sum()))
        direct=p[...,2+b].clip(1e-25);spec=p[...,6+b]/direct;dif=p[...,14+b]/direct;total=spec+dif
        cband=np.digitize(g[...,19],[.015,.06]);dbin=np.digitize(d,[60,120,180]);ebin=np.digitize(total,[.1,.5,2])
        # Match within the very same terrain/TX map and direct-geometry strata. Both arms need at least three pixels.
        groups={
         'grazing_vs_steep':(sheet[...,3]>.5,sheet[...,3]<.05,dbin*4+ebin),
         'prior_receiver_crossing_vs_first':(sheet[...,0]<.1,sheet[...,0]>.9,(dbin*4+ebin)*3+cband),
         'min_expected_below20_vs_above100':(sheet[...,8]<np.log(21),sheet[...,8]>np.log(101),dbin*4+ebin),
         'strong_vs_weak_total_extra':(total>.5,total<.1,dbin*3+cband),
         'strong_vs_weak_mirror':(spec>.3,spec<.03,(dbin*3+cband)*3+np.digitize(dif,[.05,.25])),
         'strong_vs_weak_diffuse':(dif>.15,dif<.035,(dbin*3+cband)*3+np.digitize(spec,[.05,.35]))}
        if split=='train':
            calib=clear&(g[...,15]>.5)&cal[:,None]
            cut=np.quantile(dif[calib],[.25,.75]);strength_thresholds[str(b)]=cut.tolist()
        else:cut=np.array(json.load(open(A/'diffuse_quantiles.json'))[str(b)])
        groups['diffuse_calibrated_quartiles']=(dif>cut[1],dif<cut[0],(dbin*3+cband)*3+np.digitize(spec,[.05,.35]))
        vals=np.stack((actual,y-fs,e,e*e,abs(e)>=3),2)
        contrasts={}
        for name,(high,low,key) in groups.items():
            per=np.zeros((len(ref),11));pop=clear&(g[...,15]>.5)
            for j in np.flatnonzero(~cal):
                q=pop[j];t=key[j];num=int(t.max())+1;stats=[]
                for arm in [high[j],low[j]]:
                    ii=q&arm;nn=np.bincount(t[ii],minlength=num);sums=np.stack([np.bincount(t[ii],weights=vals[j,ii,k],minlength=num) for k in range(5)],axis=1);stats.append((nn,sums/np.maximum(nn[:,None],1)))
                (a,ah),(b0,bl)=stats;wt=np.where((a>=3)&(b0>=3),np.minimum(a,b0),0)
                per[j]=np.r_[(ah*wt[:,None]).sum(0),(bl*wt[:,None]).sum(0),wt.sum()]
            z=per.sum(0);tot=z[-1];terrain=pan['terrain'][~cal];ut,inv=np.unique(terrain,return_inverse=True);by=np.zeros((len(ut),11));np.add.at(by,inv,per[~cal])
            rng=np.random.default_rng(78085);boot=by[rng.integers(len(by),size=(1000,len(by)))].sum(1);gap=(boot[:,:5]-boot[:,5:10])/np.maximum(boot[:,10:],1)
            if not tot:
                contrasts[name]=dict(status='No matched support for these fixed thresholds',matched_weight_per_arm=0,matched_maps=0);continue
            contrasts[name]=dict(status='estimated',matched_weight_per_arm=int(tot),matched_maps=int((per[:,-1]>0).sum()),column_order=['deficit3_rate','true_excess_dB','residual_bias','MSE','abs_error_ge3_rate'],
                high=(z[:5]/max(tot,1)).tolist(),low=(z[5:10]/max(tot,1)).tolist(),high_minus_low95=np.quantile(gap,[.025,.975],axis=0).T.tolist())
        band_report['matched_wide_certificate']=contrasts;allout[str(b)]=band_report
    if split=='train':
        (A/'weak_marker_calibration.json').write_text(json.dumps(oracle_weights)+'\n')
        (A/'diffuse_quantiles.json').write_text(json.dumps(strength_thresholds)+'\n')
    (A/f'{split}_matched_and_marker.json').write_text(json.dumps(allout,indent=2,allow_nan=False)+'\n')
    for b,q in allout.items():print(b,[q['wide'][k] for k in ['actual_weak_share','recall','precision','baseline_RMSE','input_marker_RMSE','label_assisted_RMSE']],q['matched_wide_certificate']['grazing_vs_steep'])
if __name__=='__main__':run(sys.argv[1])
