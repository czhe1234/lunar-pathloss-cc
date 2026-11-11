"""C42: raw9x9RXheight-difference patch on unchanged C35 FIT/C10 DEV points."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import csv,json,time
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'features/c42_rx_patch';O.mkdir(exist_ok=False)
D=R.parent/'data';rows=list(csv.DictReader((D/'train_index.csv').open()));hm=np.array([int(q['hm_row']) for q in rows]);H=np.load(D/'train_hm.npy',mmap_mode='r')
dr,dc=np.meshgrid(np.arange(-4,5),np.arange(-4,5),indexing='ij');dr=dr.reshape(1,1,81);dc=dc.reshape(1,1,81)
t0=time.time()
for part,folder in [('fit','c35_dense'),('dev','c6_points')]:
 ids=np.load(R/f'features/{folder}/{part}_map_indices.npy');pix=np.load(R/f'features/{folder}/{part}_pixel.npy').reshape(len(ids),256)
 out=np.lib.format.open_memmap(O/f'{part}_rx.npy',mode='w+',dtype=np.float32,shape=(len(ids)*256,81))
 for lo in range(0,len(ids),64):
  hi=min(lo+64,len(ids));n=hi-lo;h=np.asarray(H[hm[ids[lo:hi]]],np.float32);rr,cc=np.divmod(pix[lo:hi],256)
  ry=np.clip(rr[:,:,None]+dr,0,255);cx=np.clip(cc[:,:,None]+dc,0,255)
  base=h[np.arange(n)[:,None],rr,cc]
  patch=h[np.arange(n)[:,None,None],ry,cx]-base[:,:,None]
  out[lo*256:hi*256]=np.arcsinh(patch.reshape(-1,81)/2)/3
 out.flush();assert np.all(out[:,40]==0)
 print(json.dumps({'part':part,'rows':len(out),'seconds':time.time()-t0}),flush=True)
# FIT-only numeric statistics; retain original294stats exactly.
base=np.load(R/'audit/c34_numeric_network_20261101/normalization.npz')
x=np.load(O/'fit_rx.npy',mmap_mode='r');ids=np.linspace(0,len(x)-1,131072,dtype=np.int64)
sample=np.array(x[ids],np.float32);mean=sample.mean(0);std=sample.std(0).clip(.01);z=(sample-mean)/std
edges=np.quantile(z,np.linspace(0,1,17),axis=0).T.astype(np.float32)
a=R/'audit/c42_rx_patch_20261102';a.mkdir(exist_ok=False)
np.savez(a/'normalization.npz',mean=np.r_[base['mean'],mean],std=np.r_[base['std'],std],edges=np.concatenate([base['edges'],edges]),target_mean=base['target_mean'])
meta={'complete':True,'extra_features':81,'input':'9x9 heights relative to RX centre,asinh(delta/2)/3; edge clamp','FIT':'Same256points/map and targets as C37','DEV':'unchanged256points/map,never used to fit normalization','seconds':time.time()-t0}
(O/'complete.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
