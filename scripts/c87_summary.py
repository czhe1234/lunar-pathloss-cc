"""Cross-check denominator identities and consolidate the published-validation-only audit."""
import json,hashlib
from pathlib import Path
import numpy as np
from c87_read import ROOT,OUT,maps,write
def read(k):return json.load(open(OUT/(k+'.json')))
def main():
    full=read('full');x=np.load(OUT/'full_by_map.npz')
    vals=x['state_near_stats'];terrain=x['terrain'];tn,inv=np.unique(terrain,return_inverse=True)
    t=np.zeros((len(tn),*vals.shape[1:]));np.add.at(t,inv,vals)
    draws=np.random.default_rng(8763).integers(len(tn),size=(2000,len(tn)))
    # Keep only the statistics required for the bootstrap; avoid a giant fairway of point rows.
    comp=np.stack([t[:,:,:,:,0].sum((2,3)),t[:,:,:,:,1].sum((2,3)),
        t[:,:,0,:,1].sum(2),t[:,:,1,:,1].sum(2),t[:,:,2,:,1].sum(2),t[:,:,:,1,1].sum(2),
        t[:,:,:,:,7].sum((2,3)),t[:,:,:,:,8].sum((2,3))],-1)
    bs=comp[draws].sum(1)
    ci={}
    for i,name in enumerate(('clear','partial','blocked','near_boundary'),start=2):
        ci[name+'_share_of_total_sse']=np.quantile(bs[:,:,i]/bs[:,:,1],[.025,.975],axis=0).T.tolist()
    ci['abs_ge5_share_of_valid']=np.quantile(bs[:,:,6]/bs[:,:,0],[.025,.975],axis=0).T.tolist()
    ci['abs_ge5_share_of_total_sse']=np.quantile(bs[:,:,7]/bs[:,:,1],[.025,.975],axis=0).T.tolist()
    zs=np.load(OUT/'visibility_class.npy',mmap_mode='r');combined=np.zeros((2,3,5,2))
    yy,xx=np.indices((256,256))
    for i,row,y,p,w in maps():
        rad=np.digitize(np.hypot(xx-int(row['tx_col']),yy-int(row['tx_row'])),[50,100,150,200])
        for b in (0,1):
            good=w[b];idx=zs[i][good]*5+rad[good]
            err=(p[b].astype('float64')-y[b])[good]
            combined[b]+=np.stack([np.bincount(idx,minlength=15),np.bincount(idx,weights=err**2,minlength=15)],1).reshape(3,5,2)
    totals=combined.sum((1,2));off=full['official']
    np.testing.assert_array_equal(totals[:,0],off['valid_pixels'])
    np.testing.assert_allclose((totals[:,1]/totals[:,0])**.5,off['RMSE'],atol=1e-7,rtol=0)
    for b in (0,1):
        assert sum(full['groups'][s][b]['n'] for s in ('clear','partial','blocked'))==off['valid_pixels'][b]
        assert sum(z[0] for z in combined[b].sum(1))==off['valid_pixels'][b]
    sh=read('shift');assert all(np.array(sh['sizes']['256']['test_pixels_using_nonzero_shift'])==0)
    gr=read('geometry');assert gr['mismatches']==0 and gr['identical_sampled_visibility_values']==4771840
    w=read('weak');ex=read('examples');assert ex['clear_pair'] is not None
    out=dict(score_scope='Frozen C82 on published official fixed validation, 680 terrains, 2330 maps per band. No training, model tuning, weight/prior or submission change.',
        denominator_check=dict(counts=totals[:,0].astype(int).tolist(),sse=totals[:,1].tolist(),reconstructed_rmse=np.sqrt(totals[:,1]/totals[:,0]).tolist()),
        complete=full,official_geometry=gr,terrain_cluster_intervals=ci,frequency_and_mask=read('contrast'),
        center_horizontal_distance_bins=['<50','50-100','100-150','150-200','>=200'],
        per_band_state_distance_rmse=np.sqrt(combined[:,:,:,1]/combined[:,:,:,0]).tolist(),per_band_state_distance_n=combined[:,:,:,0].astype(int).tolist(),
        shift_controls=sh,prespecified_sample_weak_sites=w,frozen_physics_replay=read('physics'),illustrations=ex,
        output_scope_limit='C82 errors measured on released total PL; original per-ray hits, visibility and original RX interpolation weights not in release. Canonical groups are model-derived. Shift oracle and block oracles are diagnostics, not candidate scores.')
    write('summary.json',out);print('output',totals[:,0].astype(int).tolist(),np.sqrt(totals[:,1]/totals[:,0]).tolist(),flush=True)
if __name__=='__main__':main()
