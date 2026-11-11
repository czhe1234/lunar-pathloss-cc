"""Score local-shape misses independent of the 5 dB absolute amplitude definition."""
from pathlib import Path
import json
import numpy as np
from c93_score import A,ap,ratio
def main():
    q=np.load(A/'patch_target.npz');ids=q['ids'];z=np.load(A/'dense_probe/patch.npz');assert np.array_equal(z['ids'],ids)
    th=json.load(open(A/'dense_probe/choice.json'))['threshold'];arms=q['arm']
    arrays=[q[k].reshape(-1,16,7,7,2).astype('float64') for k in ['y','c82','c92']];w=q['valid'].reshape(-1,16,7,7,2)
    raw=z['pred'].reshape(-1,16,7,7,2,3);init=q['weak'].reshape(w.shape)
    masks=[];d=[[],[],[]]
    for iy in range(1,6):
        for ix in range(1,6):
            nb=w[:,:,iy-1:iy+2,ix-1:ix+2].copy();nb[:,:,1,1]=False;ww=nb.sum((2,3))>=5
            masks.append(ww&w[:,:,iy,ix])
            for k,a in enumerate(arrays):
                n=np.where(nb,a[:,:,iy-1:iy+2,ix-1:ix+2],np.nan)
                # Empty neighborhoods in masked regions do not enter the analysis.
                flat=n.reshape(len(ids),16,9,2);flat[:,:,4,:]=np.where(ww,np.nan,0.)
                d[k].append(a[:,:,iy,ix]-np.nanmedian(flat,2))
    y,c,p=[np.stack(a,axis=2) for a in d];valid=np.stack(masks,axis=2)
    hit=(y>=3)&(c<3)&(p<3)&valid
    severe=init[:,:,1:6,1:6].reshape(hit.shape)
    risk=raw[:,:,1:6,1:6,:].reshape(*hit.shape,3)
    t=arrays[0][:,:,1:6,1:6].reshape(hit.shape)
    out={}
    for name in ('uniform','grazing'):
        res=[]
        for b in (0,1):
            w=valid[:,:,:,b]&(arms==name)[:,:,None];hs=hit[:,:,:,b];rr=risk[:,:,:,b,0];call=rr>=th[b]
            true=w&hs;k=true&call;est=risk[:,:,:,b,2]
            res.append(dict(valid_centers=int(w.sum()),true_local_dips=int((w&(y[:,:,:,b]>=3)).sum()),neither_model_has_3dB_dip=int(true.sum()),
               overlap_primary=int((true&severe[:,:,:,b]).sum()),AP=ap(hs[w],rr[w]),prevalence=ratio(true.sum(),w.sum()),
               alerts=int((w&call).sum()),recall=ratio(k.sum(),true.sum()),precision=ratio(k.sum(),(w&call).sum()),event_MAE_of_primary_case_power=float(abs(est-t[:,:,:,b])[true].mean()),
               joint_within1=ratio((k&(abs(est-t[:,:,:,b])<=1)).sum(),true.sum())))
        out[name]=res
    (A/'patch_morphology_scores.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(out)
if __name__=='__main__':main()
