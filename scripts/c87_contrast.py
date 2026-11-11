"""Which dual-band differences and mask-edge regions contribute to C82 errors?"""
import numpy as np
from scipy.ndimage import binary_dilation
from c87_read import ROOT,OUT,maps,write
NAMES=['clear','partial','blocked']
def main():
    zmap=np.load(OUT/'visibility_class.npy',mmap_mode='r');joint=np.zeros((3,6,9));fill=np.zeros((2,3,2,2))
    exact145=np.zeros((2,2));fs=20*np.log10(5.775e9/415e6);x,y=np.indices((5,5))-2;disk=x*x+y*y<=4
    for i,row,y,p,w in maps():
        z=zmap[i];e=p.astype('float64')-y;mask=w.all(0);dz=(y[1].astype('float64')-y[0])[mask]-fs
        dp=(p[1].astype('float64')-p[0])[mask]-fs
        idx=z[mask].astype(int)*6+np.digitize(dz,[-5,0,5,10,20])
        vals=[np.ones(len(idx)),dz,dp,dz*dz,dp*dp,dz*dp,e[0][mask]**2,e[1][mask]**2,(dp-dz)**2]
        joint+=np.stack([np.bincount(idx,weights=a,minlength=18) for a in vals],1).reshape(3,6,9)
        for b in (0,1):
            near=binary_dilation(~w[b],structure=disk,border_value=0)&w[b]
            sel=w[b];code=z[sel].astype(int)*2+near[sel].astype(int)
            fill[b]+=np.stack([np.bincount(code,minlength=6),np.bincount(code,weights=e[b,sel]**2,minlength=6)],1).reshape(3,2,2)
            v=sel&(y[b]==145)
            exact145[b]+=[v.sum(),(e[b,v]**2).sum()]
    names=['below_minus5','minus5_to0','0_to5','5_to10','10_to20','at_least20'];vout={}
    for state,n in enumerate(NAMES):
        d=joint[state];total=d.sum(0);out={}
        for name,v in zip(names,d):
            count,zt,zp,qt,qp,prod,s0,s1,sd=v
            out[name]=dict(n=int(count),pixel_fraction=count/total[0],true_dual_excess_mean=zt/count,predicted_dual_excess_mean=zp/count,
                share_low_sse_in_state=s0/total[6],share_high_sse_in_state=s1/total[7],share_contrast_sse_in_state=sd/total[8],
                contrast_rmse=np.sqrt(sd/count))
        count,zt,zp,qt,qp,prod,s0,s1,sd=total;vy=qt/count-(zt/count)**2;vp=qp/count-(zp/count)**2
        out['total']=dict(n=int(count),actual_mean=zt/count,pred_mean=zp/count,actual_std=vy**.5,pred_std=vp**.5,
            true_pred_corr=(prod/count-zp*zt/count**2)/np.sqrt(vp*vy),contrast_rmse=np.sqrt(sd/count))
        vout[n]=out
    fout={}
    for b,band in enumerate(['415','5775']):
        d=fill[b].sum(0);den=d.sum(0)
        def stat(v,tot):
            return dict(valid_pixel_share=v[0]/tot[0],share_SSE=v[1]/tot[1],rmse=np.sqrt(v[1]/v[0]),n=int(v[0]))
        fout[band]=dict(near_invalid_mask=stat(d[1],den),beyond=stat(d[0],den),
            near_conditional_on_visibility={name:stat(fill[b,k,1],fill[b,k].sum(0)) for k,name in enumerate(NAMES)},
            valid_y_exactly_145=stat(exact145[b],den))
    write('contrast.json',dict(frequency_baseline_dB=fs,dual_band='5.775 minus 415 measured on joint valid positions, minus FSPL frequency difference',by_visibility=vout,
        mask_edge='Valid pixel with at least one invalid same-band pixel within raster Euclidean radius 2 (does not include exterior). Mask is only used in the audit.',
        mask_check=fout))
    print('blocked contrast',vout['blocked']['total'],vout['blocked']['at_least20'],'mask-near',[(b,v['near_invalid_mask']) for b,v in fout.items()],flush=True)
if __name__=='__main__':main()
