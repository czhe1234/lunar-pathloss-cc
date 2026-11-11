"""Weak-point detection and grazing controls for the frozen C82 official predictions."""
import json
import numpy as np
from scipy.spatial import cKDTree
from c87_read import ROOT,OUT,maps,write
from c87_full import boundary

C=ROOT/'audit/c78_clear'
GROUPS=['all_clear','stable','wide','switch90','intermediate']
CELLS=['true_normal_pred_normal','true_normal_pred_weak','true_weak_pred_normal','true_weak_pred_weak']
def parts(kind):
    tab={}
    for d in C.glob('val_'+kind+'_*of*'):
        a=np.load(d/('geometry.npy' if kind=='geom' else kind+'.npy'),mmap_mode='r')
        tab.update({int(i):(a,j) for j,i in enumerate(np.load(d/'positions.npy'))})
    assert len(tab)==2330
    return tab
def grazing(g,p,s,y,pred,ok,w,b):
    ratio=(p[:,6+b]+p[:,14+b])/p[:,2+b].clip(1e-25)
    bins=np.digitize(g[:,18],[60,120,180])*4+np.digitize(ratio,[.1,.5,2])
    vals=np.stack((w,(y-pred)**2,y-pred,abs(y-pred)>=5),1)
    means=[];counts=[]
    for arm in (s[:,3]>.5,s[:,3]<.05):
        ix=arm&ok;n=np.bincount(bins[ix],minlength=16)
        totals=np.stack([np.bincount(bins[ix],weights=t[ix],minlength=16) for t in vals.T],1)
        counts.append(n);means.append(totals/np.maximum(n[:,None],1))
    weight=np.where((counts[0]>=3)&(counts[1]>=3),np.minimum(*counts),0)
    return np.concatenate([(m*weight[:,None]).sum(0) for m in means]+[[weight.sum()]])
def main():
    pan=np.load(C/'val_panel.npz');ref=np.load(C/'val_reference.npy',mmap_mode='r')
    geo,pow_,sht=[parts(k) for k in ('geom','power','sheet')];zones=np.load(OUT/'visibility_class.npy',mmap_mode='r')
    grades={q['terrain']:q['combined'] for q in json.load(open(OUT/'terrains.json'))}
    lo,med,hi=np.quantile(list(grades.values()),[.25,.5,.75])
    sums=np.zeros((2330,2,len(GROUPS),4,3));gm=np.zeros((2330,2,9));allstat=np.zeros((2330,2,2))
    pair_n=0;pair_maps=[];best=None;dark=None;terrains=[]
    for i,row,y,pr,w in maps():
        terrain=row['terrain_id'];terrains.append(terrain);pix=pan['pixels'][i];r=ref[i]
        g=geo[i][0][geo[i][1]];p=pow_[i][0][pow_[i][1]];s=sht[i][0][sht[i][1]]
        vy=y.reshape(2,-1)[:,pix].astype(float);vp=pr.reshape(2,-1)[:,pix].astype(float)
        mask=w.reshape(2,-1)[:,pix];np.testing.assert_array_equal(vy.T,r[:,:2]);np.testing.assert_array_equal(mask.T,r[:,4:6]>.5)
        clear=r[:,6]>.99;stable=clear&(g[:,:11].min(1)>=.99);wide=clear&(g[:,15]>.5)
        bound=20*np.log10((4*np.pi*np.array([[415e6],[5.775e9]])/299792458)*g[:,17].clip(.001)[None])
        obs=vy-bound;pm=vp-bound;e=vp-vy
        for b in (0,1):
            weak=obs[b]>3;flag=pm[b]>3;code=weak.astype(int)*2+flag.astype(int)
            ok=mask[b];allstat[i,b]=[ok.sum(),(e[b,ok]**2).sum()]
            for j,q in enumerate([clear,stable,wide,clear&(g[:,:11].min(1)<.9),clear&(g[:,:11].min(1)>=.9)&~stable]):
                sel=q&ok;z=code[sel];ee=e[b,sel]
                sums[i,b,j]=np.stack([np.bincount(z,minlength=4),np.bincount(z,weights=ee**2,minlength=4),np.bincount(z,weights=ee,minlength=4)],1)
            gm[i,b]=grazing(g,p,s,vy[b],vp[b],wide&ok,weak,b)
        joint=mask.all(0)&wide
        weak=np.flatnonzero(joint&(obs.min(0)>8)&(pm.max(0)<1.5))
        false=np.flatnonzero(joint&(abs(obs).max(0)<1.5)&(pm.min(0)>3.5))
        xy=np.column_stack(np.divmod(pix,256));typical=lo<=grades[terrain]<=hi
        if weak.size and false.size:
            arr=cKDTree(xy[false]).query_ball_point(xy[weak],12)
            def describe(k):
                return dict(pixel=xy[k].tolist(),flat=int(pix[k]),distance_nominal=float(g[k,18]),safest_distance_upper=float(g[k,17]),
                    true_PL=vy[:,k].tolist(),pred_PL=vp[:,k].tolist(),signed_error=e[:,k].tolist(),
                    label_excess_over_conservative_direct=obs[:,k].tolist(),prediction_excess_over_same_reference=pm[:,k].tolist(),
                    min_visibility=float(g[k,:11].min()),certificate=bool(wide[k]),grazing_fraction=float(s[k,3]))
            num=sum(len(q) for q in arr)
            if num:pair_n+=num;pair_maps.append(i)
            if typical:
                for a,others in zip(weak,arr):
                    for oth in others:
                        b=false[oth];dist=np.linalg.norm(xy[a]-xy[b]);merit=abs(obs[:,a].mean()-10)+.13*dist+.4*abs(grades[terrain]-med)
                        if best is None or merit<best[0]:best=(merit,dict(index=i,sample_id=row['sample_id'],terrain=terrain,tx=[int(row['tx_row']),int(row['tx_col'])],terrain_combined_rmse=grades[terrain],positions_distance_pixels=float(dist),A=describe(a),B=describe(b)))
        if typical:
            z=zones[i];sel=(z==2)&~boundary(z)&w.all(0);er=pr-y
            ii=sel&(abs(er[0])<1)&(er[1]<-9)&(er[1]>-14)
            if ii.any():
                rr,cc=np.nonzero(ii);v=abs(-er[1,rr,cc]-11)+.3*abs(grades[terrain]-med);j=int(v.argmin())
                if dark is None or v[j]<dark[0]:
                    rc=[int(rr[j]),int(cc[j])];dark=(v[j],dict(index=i,sample_id=row['sample_id'],terrain=terrain,tx=[int(row['tx_row']),int(row['tx_col'])],pixel=rc,group='blocked beyond two pixels from boundary',terrain_combined_rmse=grades[terrain],true_PL=y[:,rc[0],rc[1]].tolist(),pred_PL=pr[:,rc[0],rc[1]].tolist(),signed_error=er[:,rc[0],rc[1]].tolist()))
    agg=sums.sum(0);tot=allstat.sum(0);mg=gm.sum(0);ut,inv=np.unique(terrains,return_inverse=True)
    by=np.zeros((len(ut),*sums.shape[1:]));np.add.at(by,inv,sums);gg=np.zeros((len(ut),2,9));np.add.at(gg,inv,gm)
    draw=np.random.default_rng(8726).integers(len(ut),size=(1500,len(ut)));gs=gg[draw].sum(1);bs=by[draw].sum(1)
    report={}
    for b,band in enumerate(('415','5775')):
        res={}
        for j,name in enumerate(GROUPS):
            m=agg[b,j];den=m.sum(0);cells={}
            for k,key in enumerate(CELLS):
                n,sse,err=m[k];cells[key]=dict(n=int(n),area_fraction=n/den[0],fraction_zone_SSE=sse/den[1],fraction_all_SSE=sse/tot[b,1],rmse=np.sqrt(sse/n),signed_error=err/n)
            boot=bs[:,b,j];br=boot[:,3,0]/(boot[:,2,0]+boot[:,3,0]);share=boot[:,2:,1].sum(1)/boot[:,:,1].sum(1)
            n=m[2:,0].sum();ss=m[2:,1].sum()
            res[name]=dict(n=int(den[0]),weak_count=int(n),weak_fraction=n/den[0],weak_share_zone_sse=ss/den[1],weak_share_all_sse=ss/tot[b,1],
                recall=m[3,0]/n,precision=m[3,0]/(m[1,0]+m[3,0]),conditional_RMSE_weak=np.sqrt(ss/n),
                recall_terrain95=np.quantile(br,[.025,.975]).tolist(),weak_SSEshare_terrain95=np.quantile(share,[.025,.975]).tolist(),cells=cells)
        m=mg[b];v=gs[:,b];delta=(v[:,:4]-v[:,4:8])/v[:,8:9]
        res['grazing_matched_wide']=dict(matched_weight_per_arm=int(m[-1]),contributing_maps=int((gm[:,b,-1]>0).sum()),
             column_order=['weak_rate','C82_MSE','label_minus_pred_bias','rate_abs_error_ge5'],
             high=(m[:4]/m[-1]).tolist(),low=(m[4:8]/m[-1]).tolist(),high_minus_low_95=np.quantile(delta,[.025,.975],axis=0).T.tolist())
        report[band]=res
    prev=json.load(open(ROOT/'audit/c79_receiver/c82_error_budget.json'))['sampled_zones']['clear_weak']
    np.testing.assert_allclose([report[b]['all_clear']['weak_share_all_sse'] for b in ('415','5775')],prev['fraction_total_SSE'],atol=1e-5)
    write('weak.json',dict(source='Official VAL only. Uniform 2048 sites per map before masking. Weak = total pathloss greater than the weakest intact direct reference plus 3 dB. True flag available for diagnosis only.',pre_mask=2330*2048,output=report))
    write('examples.json',dict(scope='Illustrative only. Pair comprised opposite mistakes at joint-valid, conservatively certified sites in the same map within 12 grid pixels. Choose a terrain in the middle half of terrain RMSE and prefer moderate 10 dB actual weak excess and a short separation.',allowed_terrain_rmse=[lo,hi],n_pairs=pair_n,n_maps=len(set(pair_maps)),n_terrains=len({terrains[i] for i in pair_maps}),clear_pair=best[1] if best else None,blocked_example=dark[1] if dark else None))
    print('sample',[(b,report[b]['all_clear']['recall'],report[b]['wide']['recall'],report[b]['all_clear']['weak_share_zone_sse']) for b in report],pair_n,None if best is None else best[1]['sample_id'],flush=True)
if __name__=='__main__':main()
