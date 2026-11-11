"""Exact input identity and target texture on the fixed16 training maps only."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path.cwd();D=R.parent/'data';A=R/'audit/c106_fit'
plan=json.loads((A/'protocol.json').read_text());ids=plan['fit16']
rows=list(csv.DictReader(open(D/'train_index.csv')));hm=np.load(D/'train_hm.npy',mmap_mode='r')
yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ww=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
index=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());lookup={i:j for j,i in enumerate(index)}
base=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
keys={};out=[]
for i in ids:
 r=rows[i];h=np.array(hm[int(r['hm_row'])]);tr=int(r['tx_row']);tc=int(r['tx_col'])
 digest=hashlib.sha256(h.tobytes()+np.array([tr,tc],np.int64).tobytes()).hexdigest()
 keys.setdefault(digest,[]).append(i);bands=[]
 for b in range(2):
  y=np.array(yy[b][i],np.float64);w=np.unpackbits(ww[b][i]).reshape(256,256).astype(bool);p=base[lookup[i],b].astype(np.float64);res=y-p
  dif=[];rdif=[]
  for axis in [0,1]:
   w0=np.take(w,range(255),axis=axis);w1=np.take(w,range(1,256),axis=axis);valid=w0&w1
   dy=np.diff(y,axis=axis)[valid];dr=np.diff(res,axis=axis)[valid];dif.append(dy);rdif.append(dr)
  dv=np.concatenate(dif);rv=np.concatenate(rdif)
  tail=w&(res>=5)
  bands.append(dict(count=int(w.sum()),C82_rmse=float(np.sqrt(np.mean(res[w]**2))),tail_count=int(tail.sum()),
   tail_sse_fraction=float((res[tail]**2).sum()/(res[w]**2).sum()),
   adjacent_label_RMS=float(np.sqrt(np.mean(dv**2))),adjacent_residual_RMS=float(np.sqrt(np.mean(rv**2))),
   residual_range=np.quantile(res[w],[0,.01,.5,.99,1]).tolist()))
 out.append(dict(mapid=i,terrain=r['terrain_id'],hm_row=int(r['hm_row']),tx=[tr,tc],input_sha256=digest,bands=bands))
dups=[v for v in keys.values() if len(v)>1]
result=dict(maps=out,exact_H_TX_duplicate_groups=dups,scope='fixed16FITmaps only; no labels used as inputs')
(A/'data_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'exactduplicateinputs':dups,'maps':[{'id':z['mapid'],'counts':[b['count'] for b in z['bands']],'C82':[b['C82_rmse'] for b in z['bands']],'residual_gradient_RMS':[b['adjacent_residual_RMS'] for b in z['bands']]} for z in out]}))
