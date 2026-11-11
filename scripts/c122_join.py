import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c122_power'
assert all((A/f'part{k}/done.json').exists() for k in range(3))
for part in ['fit','monitor']:
 n=len(np.load(R/f'audit/c121_pairs/tail_{part}/rows.npy'));seen=np.zeros(n,int)
 arrays={key:np.lib.format.open_memmap(A/f'{part}_{key}.npy',mode='w+',dtype='float32',shape=(n,59)) for key in ['clip128','wide128','wide8192']}
 for p in A.glob(f'part*/{part}_*.npz'):
  v=np.load(p);ix=v['index'];seen[ix]+=1
  for k in arrays:arrays[k][ix]=v[k]
 assert (seen==1).all()
 for a in arrays.values():a.flush()
 print(part,n,flush=True)
(A/'features_done.json').write_text(json.dumps({'labels_read':False,'all_selected_rows_exactly_once':True})+'\n')
