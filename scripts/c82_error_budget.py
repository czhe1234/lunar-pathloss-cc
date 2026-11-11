"""How much of the actual C82 error must disappear to reach one dB? Oracle arithmetic, no fitting."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';A=R/'audit/c79_receiver';C=R/'audit/c78_clear'
labels=[np.load(D/f'val_pl{s}.npy',mmap_mode='r') for s in ('415','58')]
valid=[np.load(D/f'val_mask{s}.npy',mmap_mode='r') for s in ('415','58')]
pan=np.load(C/'val_panel.npz');ref=np.load(C/'val_reference.npy',mmap_mode='r')
gdist={}
for shard in C.glob('val_geom_*of*'):
    pp=np.load(shard/'positions.npy');v=np.load(shard/'geometry.npy',mmap_mode='r')
    gdist.update({int(i):(v,j) for j,i in enumerate(pp)})
N=np.zeros(2);S=np.zeros(2);jointN=0;jointS=np.zeros(4);aloneS=np.zeros(2);aloneN=np.zeros(2)
coarse=np.zeros((2,4));rows=[]
names=['all','original_clear','other','switch','clear_weak','stable_weak','wide_weak']
ss=np.zeros((2,len(names)));nn=np.zeros_like(ss)
for p in sorted((R/'runs/c82_selected/predictions').glob('val*')):
    ii=json.load(open(p/'indices.json'))['index_rows'];pred=[np.load(p/f'pl{b}.npy',mmap_mode='r') for b in ('415','58')]
    for j,i in enumerate(ii):
        w=np.stack([np.unpackbits(a[i]).reshape(256,256).astype(bool) for a in valid])
        y=np.stack([a[i] for a in labels]).astype('float64')
        e=np.stack([a[j] for a in pred]).astype('float64')-y
        sw=w.reshape(2,-1).sum(1);se=(e**2*w).sum((1,2));N+=sw;S+=se
        joint=w[0]&w[1];v=e[:,joint];mean=(v[0]+v[1])*.5;half=(v[1]-v[0])*.5
        jointS += [v[0]@v[0],v[1]@v[1],mean@mean,half@half];jointN+=joint.sum()
        for b in (0,1):
            single=w[b]&~joint;aloneN[b]+=single.sum();aloneS[b]+=e[b,single]@e[b,single]
            for k,size in enumerate([2,4,8,16]):
                mask=w[b].reshape(256//size,size,256//size,size)
                ew=(e[b]*w[b]).reshape(mask.shape).sum((1,3));num=mask.sum((1,3))
                coarse[b,k]+=((ew**2)/np.maximum(num,1)).sum()
        xp=pan['pixels'][i];v,k=gdist[i];gd=v[k];cl=ref[i,:,6]>.99
        for b,f in enumerate([415e6,5.775e9]):
            vv=w[b].ravel()[xp];z=e[b].ravel()[xp]**2
            weak=y[b].ravel()[xp] > 20*np.log10((4*np.pi*f/299792458)*gd[:,17].clip(.001))+3
            scores=[vv,vv&cl,vv&~cl,vv&cl&(gd[:,:11].min(1)<.9),vv&cl&weak,vv&cl&(gd[:,:11].min(1)>=.99)&weak,vv&cl&(gd[:,15]>.5)&weak]
            for k,q in enumerate(scores):
                nn[b,k]+=q.sum();ss[b,k]+=z[q].sum()
out={'scope':'Actual C82 complete official fixed validation, no fitting; per-scale and common/differential decomposition over every valid pixel. Clear subsets only on the frozen uniform C78 panel of 2048 sites/map.',
     'official_valid_pixels':N.astype(int).tolist(),'official_RMSE':np.sqrt(S/N).tolist(),'required_fraction_of_SSE_to_remove_for_1dB':(1-N/S).tolist(),
     'blocks':{},'common_mode':{},'sampled_zones':{}}
for k,size in enumerate([2,4,8,16]):
    out['blocks'][str(size)]=dict(fraction_SSE_block_means=(coarse[:,k]/S).tolist(),
        RMSE_remaining_if_true_block_mean_corrected=np.sqrt((S-coarse[:,k])/N).tolist(),
        RMSE_remaining_if_all_within_block_detail_corrected=np.sqrt(coarse[:,k]/N).tolist())
assert abs((jointS[0]+jointS[1])-2*(jointS[2]+jointS[3]))<1e-4
out['common_mode']=dict(joint_valid=int(jointN),single_valid_each_band=aloneN.astype(int).tolist(),shared_fraction_of_joint_SSE=2*jointS[2]/(jointS[0]+jointS[1]),joint_baseline_RMSE=np.sqrt(jointS[:2]/jointN).tolist(),
    RMSE_after_perfect_shared_error_removal_joint_and_no_change_single=np.sqrt((jointS[3]+aloneS)/N).tolist(),
    RMSE_after_perfect_difference_removal_joint_and_no_change_single=np.sqrt((jointS[2]+aloneS)/N).tolist(),joint_difference_RMSE=2*(jointS[3]/jointN)**.5)
for k,name in enumerate(names):
    out['sampled_zones'][name]=dict(n=nn[:,k].astype(int).tolist(),fraction_valid= (nn[:,k]/nn[:,0]).tolist(),
        fraction_total_SSE=(ss[:,k]/ss[:,0]).tolist(),local_RMSE=np.sqrt(ss[:,k]/nn[:,k]).tolist(),
        full_RMSE_if_perfect_here_other_fixed=np.sqrt(np.maximum(0,ss[:,0]-ss[:,k])/nn[:,0]).tolist())
assert np.allclose(out['official_RMSE'],json.load(open(R/'audit/c82_selected_official/verified_summary.json'))['RMSE'],atol=1e-5)
(A/'c82_error_budget.json').write_text(json.dumps(out,indent=2)+'\n')
print('required',out['required_fraction_of_SSE_to_remove_for_1dB'],'4x4',out['blocks']['4'],'common',out['common_mode'],flush=True)
for name in ('original_clear','other','clear_weak'):
 print(name,out['sampled_zones'][name],flush=True)
