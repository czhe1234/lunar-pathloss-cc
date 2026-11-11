"""Which reproducible radio textures are already present in input height? TRAIN FIT only."""
import sys,json,csv
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
from c88_fingerprint import R,A,D,DIRS,shift,high,corr
def main():
    z=np.load(A/'fingerprint_train.npz');ids=z['ids'];sig=z['signature']
    rows=list(csv.DictReader(open(D/'train_index.csv')));hi=np.load(D/'train_hm.npy',mmap_mode='r')
    # Curvature retains base terrain fine texture and removes the inter-scene height offset and large crater slopes.
    out=np.zeros((2,4,4,256,256))
    for j,i in enumerate(ids):
        h=np.array(hi[int(rows[i]['hm_row'])],dtype='float32')
        for k,(dy,dx) in enumerate(DIRS):
            g=shift(h,dy,dx)-h
            for m,a in enumerate([abs(g),abs(g-uniform_filter(g,5)),abs(g-uniform_filter(g,11)),abs(shift(g,dy,dx)-g)]):
                out[j%2,m,k]+=a/(.025+uniform_filter(a,size=9))
    out/=len(ids)/2
    # Select height statistic on the first half and evaluate on the opposite terrain fold.
    train=np.array([[np.mean([corr(sig[0,b,k],out[0,m,k]) for k in range(4)]) for m in range(4)] for b in range(2)])
    best=train.argmax(1)
    cross=np.array([[corr(sig[1,b,k],out[0,best[b],k]) for k in range(4)] for b in range(2)])
    report=dict(choices=['absolute terrain gradient','5-cell high-pass gradient','11-cell high-pass gradient','second difference'],
                fit_half_matched_corr=train.tolist(),selected=best.tolist(),second_half_vs_first_half_height=cross.tolist())
    (A/'height_signature.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
