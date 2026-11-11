"""Exact released H/TX collisions and unavoidable TRAIN duplicate-label SSE."""
import csv,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path.cwd();D=R.parent/'data';A=R/'audit/c108_diagnosis';A.mkdir(exist_ok=True)
t=time.monotonic();rows=list(csv.DictReader(open(D/'train_index.csv')))
h=np.load(D/'train_hm.npy',mmap_mode='r');hashes=[hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in h]
groups=defaultdict(list)
for i,r in enumerate(rows):groups[(hashes[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col']))].append(i)
dups=[v for v in groups.values() if len(v)>1]
y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
mask=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
pop=np.array([int(i).bit_count() for i in range(256)],np.uint8)
total=[int(sum(np.sum(pop[z[j:j+512]],dtype=np.int64) for j in range(0,len(z),512))) for z in mask]
out=[];minimum=np.zeros(2);duplicate_valid=np.zeros(2,dtype=np.int64)
for ids in dups:
 bands=[]
 for b in range(2):
  yy=np.array(y[b][ids],np.float64);ww=np.unpackbits(mask[b][ids],axis=1).reshape(-1,256,256).astype(bool)
  n=ww.sum(0);s=np.where(ww,yy,0).sum(0);ss=np.where(ww,yy**2,0).sum(0)
  lower=np.maximum(ss-s*s/np.maximum(n,1),0);minimum[b]+=lower.sum();duplicate_valid[b]+=ww.sum()
  common=ww[0]&ww[1];diff=(yy[0]-yy[1])[common]
  bands.append(dict(valid=int(ww.sum()),common_firstpair=int(common.sum()),first_pair_difference_RMSE=float(np.sqrt(np.mean(diff**2))) if len(diff) else None,
   first_pair_abs_difference_quantiles=np.quantile(abs(diff),[.5,.9,.99,1]).tolist() if len(diff) else [],
   minimum_duplicate_SSE=float(lower.sum()),minimum_duplicate_RMSE=float(np.sqrt(lower.sum()/max(ww.sum(),1)))))
 out.append(dict(ids=ids,sample_id=[rows[i]['sample_id'] for i in ids],terrain=[rows[i]['terrain_id'] for i in ids],tx=[int(rows[ids[0]]['tx_row']),int(rows[ids[0]]['tx_col'])],bands=bands))
result=dict(exact_H_TX_duplicate_groups=out,group_count=len(out),maps_in_groups=sum(map(len,dups)),
 full_TRAIN_valid_pixels=total,minimum_unavoidable_duplicate_SSE=minimum.tolist(),
 full_TRAIN_RMSE_lower_bound_from_duplicates=np.sqrt(minimum/total).tolist(),
 duplicate_only_RMSE_lower_bound=np.sqrt(minimum/np.maximum(duplicate_valid,1)).tolist(),
 inference_inputs='exactreleasedHbytes+integerTXrow/col;frequencyfixed;notID',official_VAL_used=False,
 interpretation='Ifduplicateoutputsconflict,thisprovesH/TXnonidentifiabilityforthesecasesonly. Smallglobalbounddoesnotprove1dBimpossible.',
 seconds=time.monotonic()-t)
(A/'input_identity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
