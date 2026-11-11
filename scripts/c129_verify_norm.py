from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';A=R/'audit/c129_fitgap'
ys=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ms=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
result={}
for name in ['batch','global']:
 out=R/('runs/c129_norm_'+name);ids=json.loads((out/'indices.json').read_text())
 p=np.load(out/'monitor_prediction.npy',mmap_mode='r');ss=np.zeros(2);nn=np.zeros(2)
 for start in range(0,len(ids),32):
  ix=ids[start:start+32];v=np.stack([np.unpackbits(a[ix],axis=1).reshape(len(ix),256,256) for a in ms],1).astype(bool)
  y=np.stack([a[ix] for a in ys],1).astype('float64');e=np.where(v,np.asarray(p[start:start+len(ix)],dtype='float64')-y,0)
  ss+=(e*e).sum((0,2,3));nn+=v.sum((0,2,3))
 z=dict(RMSE=np.sqrt(ss/nn).tolist(),valid=nn.astype(int).tolist())
 old=json.loads((out/'result.json').read_text())['monitor']
 assert np.max(np.abs(np.array(z['RMSE'])-old['rmse']))<1e-5
 result[name]=z
result['status']='passed';result['source']='originalTRAINlabels/packedmasks/NumPyfloat64';result['official_val_used']=False
(A/'norm_verified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
