from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver'
p=json.load(open(A/'protocol.json'))['panel'];ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r')
rs=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
hh=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');yy=np.load(R.parent/'data/train_pl415.npy',mmap_mode='r');mm=np.load(R.parent/'data/train_mask415.npy',mmap_mode='r')
d0=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1';base=np.load(d0/'pl415.npy',mmap_mode='r')
local=np.load(R/'runs/c72_observed/full_internal/correction.npy',mmap_mode='r')
graph=np.load(R/'runs/c73_wide_real/full_internal/prediction.npy',mmap_mode='r')
cal=[r for r in p if r['group']=='fit'];rough=[np.std(hh[int(rs[r['index']]['hm_row'])]) for r in cal]
chosen=[cal[i] for i in np.argsort(rough)[[10,40,60]]]
fig,ax=plt.subplots(3,5,figsize=(14,8.5),dpi=103)
Y,X=np.indices((256,256));ix=np.load(R/'src/c70_centroid_lookup0.npz')
for i,r in enumerate(chosen):
 j,k=r['pos'],r['index'];q=rs[k];h=hh[int(q['hm_row'])].astype(float);tr,tc=int(q['tx_row']),int(q['tx_col']);z=np.floor(h[min(tr+1,255),tc])+3
 fs=20*np.log10(4*np.pi*415e6/299792458*np.sqrt((X-tc)**2+(Y-tr)**2+(h+1-z)**2).clip(.25))
 y=yy[k].astype(float);m=np.unpackbits(mm[k]).reshape(256,256).astype(bool);pr=.5*(base[j]+local[j,0]+graph[j,0])
 sy,sx=np.gradient(h);d=np.sqrt((X-tc)**2+(Y-tr)**2+(h+1-z)**2).clip(.25)
 cos=(h+1-z-sy*(Y-tr)-sx*(X-tc))/(d*np.sqrt(1+sx**2+sy**2))
 for v,a,title,lo,hi,cmap in zip([h,(y-fs),(pr-fs),(y-pr),-np.log10(abs(cos).clip(1e-5))],ax[i],['height m','label minus FSPL','C76 minus FSPL','residual dB','-log abs cos'],[h.min(),-2,-2,-6,0],[h.max(),16,16,6,3],['terrain','magma','magma','RdBu_r','viridis']):
  view=v.copy()
  if title in ('label minus FSPL','residual dB'):view[~m]=np.nan
  a.imshow(view,cmap=cmap,vmin=lo,vmax=hi);a.plot(tc,tr,'gx',ms=5);a.set_xticks([]);a.set_yticks([]);a.set_title(title+' '+r['terrain'] if title=='height m' else title,fontsize=9)
fig.tight_layout();fig.savefig(A/'fit_cases.jpg')
