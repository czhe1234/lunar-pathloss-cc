"""Sampled geometry/target alignment diagnostic; does not fit a predictor."""
from pathlib import Path
import csv,json
import numpy as np
root=Path(__file__).resolve().parents[1];data=root.parent/'data';rng=np.random.default_rng(4871);out={}
for split,n in [('train',256),('val',64)]:
 rows=list(csv.DictReader((data/f'{split}_index.csv').open())); ids=sorted(map(int,rng.choice(len(rows),n,replace=False)))
 for band in ['415','58']:
  a=np.load(data/f'{split}_pl{band}.npy',mmap_mode='r',allow_pickle=False);p=np.load(data/f'{split}_mask{band}.npy',mmap_mode='r',allow_pickle=False)
  d={k:[] for k in ['as_index','flip_row','flip_col','transpose']};bit_checks=True
  for i in ids:
   m=np.unpackbits(p[i],bitorder='big').reshape(256,256).astype(bool); arr=a[i].astype(np.float32)
   j=int(np.argmin(np.where(m,arr,np.inf)));rr,cc=divmod(j,256);tr=int(rows[i]['tx_row']);tc=int(rows[i]['tx_col'])
   for k,(r,c) in {'as_index':(tr,tc),'flip_row':(255-tr,tc),'flip_col':(tr,255-tc),'transpose':(tc,tr)}.items():d[k].append(float(np.hypot(rr-r,cc-c)))
   for k in [0,1,7,8,32767,32768,65535]:bit_checks &= bool(m.reshape(-1)[k])==bool((int(p[i,k//8])>>(7-k%8))&1)
   bit_checks &= bool(np.array_equal(np.packbits(m.reshape(-1)),p[i]))
  out[f'{split}_{band}']={'sample_count':n,'index_rows':ids,'mask_bit_checks_pass':bool(bit_checks),'distance_min_valid_PL_pixel_to_TX_m':{k:{'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'within_3m_fraction':float((np.asarray(v)<=3).mean())} for k,v in d.items()}}
(root/'audit'/'input_alignment.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{j:v[j] for j in ['sample_count','mask_bit_checks_pass','distance_min_valid_PL_pixel_to_TX_m']} for k,v in out.items()},ensure_ascii=False))
