"""Independent finite launch-count controls at actual receiving facets. No labels, masks or earlier PL maps."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));from src.c66_lattice import counts
A=R/'audit/c93_joint_miss';D=R.parent/'data'
def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=['screen','confirm','patch']);a=p.parse_args()
    inp=np.load(A/(a.split+'_features.npz'));ids=inp['ids'];pixels=inp['pixels']
    rows=list(csv.DictReader(open(D/'train_index.csv')));h=np.load(D/'train_hm.npy',mmap_mode='r');lookup=np.load(R/'src/c70_centroid_lookup0.npz')
    faces=lookup['ids'];weights=lookup['weights'];axis=np.linspace(-128,128,256,dtype='float32')
    iy,ix=np.indices((256,256));xy=np.stack([axis[ix].ravel(),-axis[iy].ravel()],1)
    a0=(iy[:-1,:-1]*256+ix[:-1,:-1]).ravel();fcs=np.concatenate([np.stack([a0,a0+1,a0+257],1),np.stack([a0,a0+257,a0+256],1)],0)
    data=[];ts=time.monotonic()
    for j,i in enumerate(ids):
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);hh=h[int(row['hm_row'])]
        face,inv=np.unique(faces[pixels[j]],return_inverse=True);inv=inv.reshape(-1,3);w=weights[pixels[j]]
        vertices=fcs[face];pts=np.concatenate([xy[vertices],(hh.ravel()[vertices]+1)[:,:,None]],2).astype('float64')
        pts-=np.array([tc-128.,127.-tr,np.floor(hh[min(tr+1,255),tc])+3.])
        tmp=[]
        for angle in [0.,.013,-.017]:
            vv=pts.copy();ca,sa=np.cos(angle),np.sin(angle);vv[:,:,0]=pts[:,:,0]*ca+pts[:,:,1]*sa;vv[:,:,1]=pts[:,:,1]*ca-pts[:,:,0]*sa
            raw=counts(vv,4_000_000_000,128)
            found=(raw[:,2]>=0)[inv];n=raw[:,2][inv];ratio=raw[:,0][inv];size=raw[:,1][inv]
            # High expected count/non-enumerated faces retain unit continuum density; coverage is explicit.
            assert np.isfinite(ratio).all() and (ratio>=0).all()
            coverage=(w*found).sum(1);empty=(w*(n==0)).sum(1)
            direct=(w*ratio).sum(1)
            tmp.append(np.stack([coverage,empty,-10*np.log10(direct.clip(.0001,1e4)),(w*(-10*np.log10(ratio.clip(.0001,1e4)))).sum(1),
                (w*np.log1p(size.clip(0,1e8))).sum(1),np.sum(w*(ratio<.4),axis=1)],1))
        data.append(np.stack(tmp,1).astype('float32'))
        if (j+1)%100==0:print(j+1,len(ids),'seconds',round(time.monotonic()-ts,1),flush=True)
    np.savez_compressed(A/(a.split+'_phase.npz'),ids=ids,pixels=pixels,data=np.stack(data),angles=[0.,.013,-.017])
    print('done',len(ids),time.monotonic()-ts)
if __name__=='__main__':main()
