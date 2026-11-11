from pathlib import Path
import json,csv,time
import numpy as np
from scipy.ndimage import maximum_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c119_polarity';P=R/'audit/c101_residual/monitor'
ids=json.loads((P/'done.json').read_text())['ids'];px=np.load(P/'pixel.npy').reshape(len(ids),-1)
index={i:j for j,i in enumerate(json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'])}
field=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
ys=[np.load(R.parent/'data'/('train_pl'+b+'.npy'),mmap_mode='r') for b in ['415','58']]
ws=[np.load(R.parent/'data'/('train_mask'+b+'.npy'),mmap_mode='r') for b in ['415','58']]
count=np.zeros(2);sse=np.zeros(2);weak=np.zeros(2);wc={r:np.zeros(2) for r in [1,2,4]};cs={r:np.zeros(2) for r in wc};ec={r:np.zeros(2) for r in wc}
for j,i in enumerate(ids):
 for b in range(2):
  m=np.unpackbits(ws[b][i]).reshape(256,256).astype(bool);y=ys[b][i].astype('float32');res=y-field[index[i],b];pos=m&(res>=5);neg=m&(res<=-5);q=px[j]
  w=m.ravel()[q];e=res.ravel()[q].astype('float64');pp=pos.ravel()[q]
  count[b]+=w.sum();sse[b]+=(e*e*w).sum();weak[b]+=pp.sum()
  for r in wc:
   pn=maximum_filter(neg,size=2*r+1,mode='constant');np_=maximum_filter(pos,size=2*r+1,mode='constant')
   eligible=(pos&pn)|(neg&np_);v=eligible.ravel()[q]
   wc[r][b]+=(pp&pn.ravel()[q]).sum();cs[r][b]+=(e*e*v).sum();ec[r][b]+=v.sum()
out=dict(scope='128monitor same2048sampledpixels;true-label-assisted loss-scope diagnostic,NOTprediction',base_rmse=np.sqrt(sse/count).tolist(),
 radii={r:dict(weak_neighbor_fraction=(wc[r]/weak).tolist(),eligible_pixel_fraction=(ec[r]/count).tolist(),eligible_SSE_fraction=(cs[r]/sse).tolist(),perfect_pair_endpoints_remaining_RMSE=np.sqrt((sse-cs[r])/count).tolist()) for r in wc},official_val_used=False)
(A/'scope.json').write_text(json.dumps(out,indent=2));print(out,flush=True)
