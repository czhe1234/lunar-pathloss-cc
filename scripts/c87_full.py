"""Exact masked error concentration and within-block texture on frozen validation maps."""
import json,time
import numpy as np
from scipy.ndimage import distance_transform_edt
from c87_read import ROOT,DATA,OUT,maps,write

STATES=['clear','partial','blocked']
THRESH=(3,5,10,20)
FIELDS=['n','sse','sum_error','sum_abs','sse_too_bright']
for t in THRESH:FIELDS.extend([f'n_ge{t}',f'sse_ge{t}'])
K=len(FIELDS)
def accumulate(ix,e,n):
    ae=np.abs(e);sq=e*e
    vec=[np.bincount(ix,minlength=n),np.bincount(ix,weights=sq,minlength=n),
         np.bincount(ix,weights=e,minlength=n),np.bincount(ix,weights=ae,minlength=n),
         np.bincount(ix,weights=sq*(e<0),minlength=n)]
    for t in THRESH:
        sel=ae>=t
        vec.extend([np.bincount(ix[sel],minlength=n),np.bincount(ix[sel],weights=sq[sel],minlength=n)])
    return np.array(vec).T
def summarize(a,tot):
    n,se,e,ab,neg=a[:5]
    if not n:return dict(n=0)
    r=dict(n=int(n),pixel_share=n/tot[0],sse_share=se/tot[1],rmse=(se/n)**.5,
           mean_error_pred_minus_label=e/n,mae=ab/n,sse_power_overpredicted_fraction=neg/se if se else None)
    for i,t in enumerate(THRESH):
        cnt,sst=a[5+2*i:7+2*i]
        r['abs_ge'+str(t)]=dict(n=int(cnt),pixel_fraction_in_group=cnt/n,fraction_of_group_SSE=sst/se if se else None,fraction_of_total_SSE=sst/tot[1])
    return r
def boundary(z):
    edge=z==1
    d=z[:,1:]!=z[:,:-1];edge[:,1:]|=d;edge[:,:-1]|=d
    d=z[1:]!=z[:-1];edge[1:]|=d;edge[:-1]|=d
    return distance_transform_edt(~edge)<=2 if edge.any() else np.zeros_like(edge)
def texture(y,p,w,sz):
    n=len(y);shape=(n,256//sz,sz,256//sz,sz)
    count=w.reshape(shape).sum((2,4)).clip(1)
    sy=(y*w).reshape(shape).sum((2,4));sp=(p*w).reshape(shape).sum((2,4))
    # Within-block variations around the respective masked means. Center first to avoid cancellation.
    dy=(y.reshape(shape)-sy[:,:,None,:,None]/count[:,:,None,:,None])*w.reshape(shape)
    dp=(p.reshape(shape)-sp[:,:,None,:,None]/count[:,:,None,:,None])*w.reshape(shape)
    axis=(1,2,3,4)
    return np.stack([(dy*dy).sum(axis),(dp*dp).sum(axis),(dy*dp).sum(axis)],1)
def main():
    while not (OUT/'geometry.json').exists():time.sleep(1)
    zmap=np.load(OUT/'visibility_class.npy',mmap_mode='r')
    sums=np.zeros((len(zmap),2,3,2,K))
    joint=np.zeros((len(zmap),3,7));onevalid=np.zeros((2,K));rad=np.zeros((2,5,2));hist=np.zeros((2,2401),np.int64)
    textures=np.zeros((len(zmap),2,3,3));rows=[];t0=time.monotonic()
    yy,xx=np.indices((256,256));sizes=[2,4,8]
    for i,row,y,p,w in maps():
        z=zmap[i];near=boundary(z);err=p.astype('float64')-y
        radial=np.digitize(np.hypot(xx-int(row['tx_col']),yy-int(row['tx_row'])),[50,100,150,200])
        for b in (0,1):
            mask=w[b];e=err[b][mask];kk=(z[mask]*2+near[mask])
            sums[i,b]=accumulate(kk,e,6).reshape(3,2,K)
            ae=np.abs(e);hist[b]+=np.bincount(np.minimum(2400,(ae/.025).astype(int)),minlength=2401)
            rr=radial[mask];rad[b,:,0]+=np.bincount(rr,minlength=5);rad[b,:,1]+=np.bincount(rr,weights=e*e,minlength=5)
            single=w[b]&~w[1-b];es=err[b][single]
            if es.size:onevalid[b]+=accumulate(np.zeros(es.size,np.uint8),es,1)[0]
        j=w.all(0);lo=err[0,j];hi=err[1,j];jz=z[j];delta=(hi-lo)/2;mean=(hi+lo)/2
        joint[i]=np.stack([np.bincount(jz,weights=a,minlength=3) for a in (np.ones(len(lo)),lo*lo,hi*hi,lo,hi,lo*hi,delta*delta)],1)
        for k,sz in enumerate(sizes):textures[i,:,k]=texture(y.astype('float64'),p.astype('float64'),w,sz)
        rows.append((i,row['terrain_id'],row['sample_id'],row['group'],int(row['hm_row'])))
        if (i+1)%600==0:print(i+1,'maps','seconds',round(time.monotonic()-t0,1),flush=True)
    rows.sort();assert [r[0] for r in rows]==list(range(len(zmap)))
    terrain=np.array([r[1] for r in rows]);names,inv=np.unique(terrain,return_inverse=True);ut,first=np.unique(terrain,return_index=True)
    assert len(names)==680 and len(rows)==2330
    official=json.load(open(ROOT/'audit/c82_selected_official/verified_summary.json'))
    base=np.load(ROOT/'runs/c82_selected/per_map_val.npz');by_map=sums.sum((2,3))
    np.testing.assert_array_equal(by_map[base['ids'],:,0],base['count'])
    np.testing.assert_allclose(by_map[base['ids'],:,1],base['sse'],rtol=1e-10,atol=1e-5)
    byter=np.zeros((len(names),2,3,2,K));np.add.at(byter,inv,sums);tm=byter.sum((2,3))
    rm=np.sqrt(tm[:,:,1]/tm[:,:,0]);terrain_score=rm.mean(1)
    heights=np.load(DATA/'val_hm.npy',mmap_mode='r');rough=[];relief=[]
    for i in first:
        h=heights[rows[i][4]].astype(float);sy,sx=np.gradient(h)
        rough.append(np.sqrt(np.mean(sx*sx+sy*sy)));relief.append(np.ptp(h))
    total=tm.sum(0);over=sums.sum(0)
    def per_band(v):return [summarize(v[b],total[b]) for b in (0,1)]
    group=dict(all=per_band(total),near_boundary=per_band(over[:,: ,1].sum(1)),beyond_2m=per_band(over[:,: ,0].sum(1)))
    for code,name in enumerate(STATES):
        group[name]=per_band(over[:,code].sum(1))
        for n,tag in [(0,'interior'),(1,'near')]:group[name+'_'+tag]=per_band(over[:,code,n])
    group['single_valid_only']=per_band(onevalid)
    for gp in ('single','multi'):
        ix=[r[0] for r in rows if r[3]==gp];group[gp+'_terrain_group']=per_band(by_map[ix].sum(0))
    cuts=np.quantile(rough,[.25,.5,.75]);bins=np.digitize(rough,cuts)
    for k in range(4):group['roughness_q'+str(k+1)]=per_band(tm[bins==k].sum(0))
    jrows=joint.sum(0);both={}
    for name,j in [('all',jrows.sum(0)),*zip(STATES,jrows)]:
        n,s0,s1,l,h,x,d=j
        denom=np.sqrt((s0-l*l/n)*(s1-h*h/n));ss=s0+s1
        both[name]=dict(n=int(n),rmse=np.sqrt([s0/n,s1/n]).tolist(),signed_residual_correlation=(x-l*h/n)/denom,
                        shared_fraction_of_joint_sse=1-2*d/ss,difference_rmse=2*np.sqrt(d/n))
    texture_scores={};tt=textures.sum(0)
    for k,sz in enumerate(sizes):
        these=[]
        for b in (0,1):
            vy,vp,cross=tt[b,k];n,se=total[b,:2];detail=vy+vp-2*cross
            state=dict(actual_detail_rms=(vy/n)**.5,predicted_detail_rms=(vp/n)**.5,amplitude_ratio=(vp/vy)**.5,
                       pattern_correlation=cross/np.sqrt(vy*vp),fraction_total_sse_in_detail=detail/se,
                       best_single_detail_multiplier=cross/vp,
                       rmse_if_only_global_detail_amplitude_corrected=np.sqrt(max(0,se-(vp-cross)**2/vp)/n),
                       remaining_rms_if_true_means_and_oracle_detail_multiplier=np.sqrt(max(0,vy-cross*cross/vp)/n))
            these.append(state)
        texture_scores[str(sz)]=these
    worst=np.argsort(-terrain_score);n10=int(len(names)*.1)
    def ranked_by(ix):
        losing=tm[ix].sum(0);rest=total-losing
        return dict(terrains=len(ix),maps=int(np.isin(inv,ix).sum()),fraction_pixels=(losing[:,0]/total[:,0]).tolist(),
                    fraction_sse=(losing[:,1]/total[:,1]).tolist(),excluded_remaining_RMSE=np.sqrt(rest[:,1]/rest[:,0]).tolist())
    concentration={'by_worst_combined_terrain_RMSE_top68':ranked_by(worst[:n10]),
                   'by_total_sse_top68':ranked_by(np.argsort(-tm[:,:,1].sum(1))[:n10]),
                   'terrain_rmse_quantiles':np.quantile(rm,[.1,.5,.9,.95,.99],axis=0).tolist(),
                   'terrain_RMSE_both_le1':int((rm<=1).all(1).sum()),'terrain_RMSE_le1_each':(rm<=1).sum(0).tolist(),
                   'n_multi':int(sum(rows[i][3]=='multi' for i in first))}
    table=[]
    for k,t in enumerate(names):
        table.append(dict(terrain=str(t),maps=int((inv==k).sum()),valid=tm[k,:,0].astype(int).tolist(),rmse=rm[k].tolist(),combined=terrain_score[k],
                     fraction_total_sse=(tm[k,:,1]/total[:,1]).tolist(),gradient_rms=rough[k],height_range_m=relief[k]))
    permapbias=(by_map[:,:,2]**2/by_map[:,:,0]).sum(0);bias=dict(global_mean_error=(total[:,2]/total[:,0]).tolist(),
        global_constant_oracle_RMSE=np.sqrt(total[:,1]/total[:,0]-(total[:,2]/total[:,0])**2).tolist(),
        per_map_constant_oracle_RMSE=np.sqrt((total[:,1]-permapbias)/total[:,0]).tolist(),
        fraction_total_sse_map_offsets=(permapbias/total[:,1]).tolist())
    q={}
    for k in (.5,.9,.95,.99):
        q[str(k)]=[(np.searchsorted(hist[b].cumsum(),k*total[b,0])*.025) for b in (0,1)]
    distance={key:[dict(n=int(rad[b,i,0]),rmse=np.sqrt(rad[b,i,1]/rad[b,i,0]),share_sse=rad[b,i,1]/total[b,1]) for b in (0,1)]
              for i,key in enumerate(['<50','50-100','100-150','150-200','>=200'])}
    out=dict(official=official,definition='error=predicted pathloss minus label; negative means predicted received power too strong. Shares relative to total masked SSE in the same band.',
             map_bias=bias,groups=group,frequency_joint=both,texture=texture_scores,abs_error_quantile_bin_lower_edge_db=q,quantile_bin_width=.025,
             center_horizontal_distance_m=distance,roughness_quartile_bounds=cuts.tolist(),terrain_concentration=concentration,
             worst_terrain_examples=[table[k] for k in worst[:7]],seconds=time.monotonic()-t0)
    np.savez_compressed(OUT/'full_by_map.npz',state_near_stats=sums,fields=FIELDS,texture=textures,joint=joint,terrain=terrain)
    write('full.json',out);write('terrains.json',table)
    print('groups',(total[:,1]/total[:,0])**.5,{k:[(round(r['pixel_share'],4),round(r['sse_share'],4),round(r['rmse'],3)) for r in group[k]] for k in ('clear','partial','blocked','near_boundary','blocked_interior')},flush=True)
if __name__=='__main__':main()
