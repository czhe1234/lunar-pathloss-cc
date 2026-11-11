"""Identical published heightmap+TX audit. CPU only; no model fitting.
A disagreement at exactly identical released input is an observability witness.
The empirical replicate bound is not generalized to nonreplicated scenes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'audit/c18_identifiability_20261101';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((D/'train_index.csv').open()));h=np.load(D/'train_hm.npy',mmap_mode='r')
start=time.time();digests=[];hg=defaultdict(list)
for i in range(len(h)):
 v=np.ascontiguousarray(h[i],dtype=np.float32);key=hashlib.sha256(v.tobytes()).hexdigest();digests.append(key);hg[key].append(i)
groups=defaultdict(list)
for i,r in enumerate(rows):groups[(digests[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']))].append(i)
repeats=[ii for ii in groups.values() if len(ii)>1]
y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
m=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
records=[]
for ii in repeats:
 r=rows[ii[0]];rec={'index_rows':ii,'sample_ids':[rows[i]['sample_id'] for i in ii],'terrain_ids':[rows[i]['terrain_id'] for i in ii],'TX_row_col':[int(r['tx_row']),int(r['tx_col'])],'bands':{}}
 for k,b in enumerate(['415','58']):
  yy=np.array(y[k][ii],dtype=np.float64);mm=np.unpackbits(m[k][ii],axis=1,bitorder='big').reshape(-1,256,256).astype(bool)
  count=mm.sum(0);mean=(yy*mm).sum(0)/np.maximum(count,1);sse=((yy-mean)**2*mm).sum(0);rep=count>=2
  pairs=[]
  for a in range(len(ii)):
   for z in range(a+1,len(ii)):
    both=mm[a]&mm[z];err=yy[a][both]-yy[z][both]
    pairs.append({'rows':[ii[a],ii[z]],'joint_valid':int(both.sum()),'RMSE':float(np.sqrt(np.mean(err**2))) if len(err) else None,'max_abs':float(np.max(abs(err))) if len(err) else None,'different_valid_labels':int(np.count_nonzero(err))})
  rec['bands'][b]={'replicate_pixel_observations':int(count[rep].sum()),'best_constant_per_input_SSE':float(sse[rep].sum()),'best_constant_RMSE':float(np.sqrt(sse[rep].sum()/count[rep].sum())) if rep.any() else None,'pairs':pairs}
 records.append(rec)
out={'scope':'All official TRAIN inputs, no official VAL and no model training','maps':len(rows),'heightmaps':len(h),'unique_heightmaps':len(hg),'duplicate_heightmap_rows':[v for v in hg.values() if len(v)>1],'identical_H_TX_groups':len(repeats),'maps_in_repeated_groups':sum(map(len,repeats)),'records':records,'seconds':time.time()-start,'interpretation':'Only byte-identical normalized float32 H and identical integer TX count. Different labels establish missing observation information or label inconsistency for those repeated inputs. No repeats, or equal labels, cannot prove all inputs sufficient.'}
(O/'identical_input_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
