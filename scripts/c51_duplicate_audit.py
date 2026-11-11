"""Check exact repeated H/TX inputs against released labels, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
rows=list(csv.DictReader((D/'train_index.csv').open()));groups=defaultdict(list)
for i,r in enumerate(rows):groups[(r['hm_row'],r['tx_row'],r['tx_col'])].append(i)
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
H=np.load(D/'train_hm.npy',mmap_mode='r')
sums=np.zeros(2);counts=np.zeros(2,dtype=np.int64);pairs=[]
for key,ids in groups.items():
 if len(ids)<2:continue
 assert len(ids)==2
 i,j=ids;rm=[];ss=[];cc=[];fraction=[]
 for b in range(2):
  keep=np.unpackbits(M[b][i],bitorder='big').astype(bool)&np.unpackbits(M[b][j],bitorder='big').astype(bool)
  a=Y[b][i].reshape(-1)[keep].astype(np.float64);z=Y[b][j].reshape(-1)[keep].astype(np.float64)
  e=a-z;s=float(e@e);c=int(keep.sum());sums[b]+=s;counts[b]+=c
  rm.append(float(np.sqrt(s/c)));ss.append(s);cc.append(c);fraction.append(float((a==z).mean()))
 pairs.append({'indices':ids,'sample_ids':[rows[k]['sample_id'] for k in ids],'terrain':rows[i]['terrain_id'],
  'input_key':key,'heightmap_sha256':hashlib.sha256(np.asarray(H[int(key[0])]).tobytes()).hexdigest(),
  'RMSE_label_difference':rm,'SSE':ss,'joint_valid_count':cc,'fraction_labels_bitwise_equal':fraction})
out={'scope':'Exact same released H array and integerTX pixel; officialTRAIN only',
 'pair_count':len(pairs),'terrain_count':len({p['terrain'] for p in pairs}),'pairs':pairs,
 'pooled_pair_difference_RMSE':np.sqrt(sums/counts).tolist(),'joint_valid_count':counts.tolist(),
 'best_shared_prediction_RMSE_on_these_joint_pixels':np.sqrt(sums/counts/4).tolist(),
 'meaning':'These repeated inputs have nearly consistent labels; no multi-dB target ambiguity observed in this subset.',
 'limits':'Repeated runs are not established independent solver seeds. This does not estimate independent MonteCarlo noise or prove a global official-VAL error floor. It does not recover the raw simulation mesh.'}
A=R/'audit/c51_duplicate_inputs_20261102';A.mkdir(exist_ok=True);(A/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['pair_count','terrain_count','pooled_pair_difference_RMSE','joint_valid_count','limits']}))
