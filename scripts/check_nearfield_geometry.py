"""Near-source label normalization on flat fit-terrain patches, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT.parent/'data';O=ROOT/'audit/geometry_model_diagnosis_20261031'
rows=list(csv.DictReader((D/'train_index.csv').open()));dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids']);H=np.load(D/'train_hm.npy',mmap_mode='r');Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')];lam=np.array([299792458/415e6,299792458/5775e6])
selected=[]
for i,r in enumerate(rows):
 tr=int(r['tx_row']);tc=int(r['tx_col'])
 if r['terrain_id'] in dev or min(tr,tc)<8 or max(tr,tc)>247:continue
 h=H[int(r['hm_row'])]
 if np.ptp(h[tr-4:tr+5,tc-4:tc+5])<=.25:selected.append(i)
dr,dc=np.meshgrid(np.arange(-4,5),np.arange(-4,5),indexing='ij');dr=dr.ravel();dc=dc.ravel();radius=np.hypot(dr,dc)
result={}
for txh in (2.,3.):
 for shift in [(0.,0.),(.5,-.5)]:
  key=f'tx{txh}_rx1_txoffset{shift}';result[key]={}
  for i in selected:
   r=rows[i];tr=int(r['tx_row']);tc=int(r['tx_col']);h=H[int(r['hm_row'])];dz=h[tr+dr,tc+dc]+1.-(float(h[tr,tc])+txh);dist=np.sqrt((dr-shift[0])**2+(dc-shift[1])**2+dz**2)
   ref=20*np.log10(4*np.pi*dist[None]/lam[:,None])
   for k,b in enumerate(('415','58')):
    valid=np.unpackbits(M[k][i],bitorder='big').reshape(256,256)[tr+dr,tc+dc].astype(bool);excess=np.asarray(Y[k][i,tr+dr,tc+dc],np.float64)-ref[k]
    for lo,hi in [(0,1.01),(1.01,2.01),(2.01,3.01),(3.01,5.7)]:
     q=(radius>=lo)&(radius<hi)&valid;x=excess[q];a=result[key].setdefault(b+f'_r{lo}_{hi}',{'n':0,'sum':0.,'square':0.});a['n']+=len(x);a['sum']+=float(x.sum());a['square']+=float((x*x).sum())
  for v in result[key].values():
   v['mean_label_minus_reference_db']=v['sum']/v['n'] if v['n'] else None;v['std_db']=math.sqrt(max(0,v['square']/v['n']-(v['sum']/v['n'])**2)) if v['n'] else None
out={'scope':'fit terrains only, local9x9height range<=0.25m, no geometry-test development terrains; diagnostic normalization, not new propagation targets','samples':len(selected),'terrains':len({rows[i]['terrain_id'] for i in selected}),'original_rows':selected,'results':result,'limits':'Finite triangle-area averaging and interpolation alter near-source values. This alone cannot recover actual TX height. No material/path simulation performed.'}
(O/'nearfield_geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='original_rows'}),flush=True)
