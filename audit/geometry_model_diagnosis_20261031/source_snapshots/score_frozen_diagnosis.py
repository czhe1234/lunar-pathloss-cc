"""Independent float64 masked score and residual scale decomposition. No fit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/geometry_model_diagnosis_20261031';D=ROOT.parent/'data';F=OUT/'frozen_predictions'
assert (F/'inference.json').exists()
idx=json.loads((F/'indices.json').read_text());rows=idx['index_rows'];parts=idx['parts'];table=list(csv.DictReader((D/'train_index.csv').open()))
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
ps={n:np.load(F/(n+'.npy'),mmap_mode='r') for n in ('b0_bf16','b0_fp32','b1_bf16','b1_fp32')}
old=ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16';oldidx={i:j for j,i in enumerate(json.loads((old/'indices.json').read_text())['index_rows'])};oldp=[np.load(old/f'pl{b}.npy',mmap_mode='r') for b in ('415','58')]
agg={};permap=[];diff={};replay={'max_abs':0.,'square':0.,'n':0};start=time.time()
for j,(i,part) in enumerate(zip(rows,parts)):
 for k,b in enumerate(('415','58')):
  y=Y[k][i].astype(np.float64);v=np.unpackbits(M[k][i],bitorder='big').reshape(256,256).astype(bool);n=int(v.sum())
  for name,p in ps.items():
   e=np.where(v,p[j,k].astype(np.float64)-y,0);sse=float((e*e).sum());a=agg.setdefault(name,{}).setdefault(part,{}).setdefault(b,{'n':0,'sse':0.,'sum_error':0.,'scale_sse':{str(s):0. for s in (2,4,8,16,32,64,128,256)}});a['n']+=n;a['sse']+=sse;a['sum_error']+=float(e.sum())
   for s in (2,4,8,16,32,64,128,256):
    es=e.reshape(256//s,s,256//s,s).sum((1,3));ns=v.reshape(256//s,s,256//s,s).sum((1,3));a['scale_sse'][str(s)]+=float(np.divide(es*es,ns,out=np.zeros_like(es),where=ns>0).sum())
   permap.append({'prediction_row':j,'index_row':i,'terrain_id':table[i]['terrain_id'],'part':part,'run':name,'band':b,'n':n,'sse':sse})
  for model in ('b0','b1'):
   d=(ps[model+'_bf16'][j,k].astype(np.float64)-ps[model+'_fp32'][j,k].astype(np.float64))[v];a=diff.setdefault(model,{}).setdefault(part,{}).setdefault(b,{'n':0,'square':0.,'max_abs':0.});a['n']+=len(d);a['square']+=float(d@d);a['max_abs']=max(a['max_abs'],float(np.abs(d).max()))
  if part=='dev':
   d=ps['b0_bf16'][j,k].astype(np.float64)-oldp[k][oldidx[i]].astype(np.float64);replay['max_abs']=max(replay['max_abs'],float(abs(d).max()));replay['square']+=float((d*d).sum());replay['n']+=d.size
for name,runs in agg.items():
 for part,bands in runs.items():
  for b,a in bands.items():
   a['rmse']=(a['sse']/a['n'])**.5;a['bias']=a['sum_error']/a['n'];a['scale_sse_share']={s:x/a['sse'] for s,x in a['scale_sse'].items()}
for runs in diff.values():
 for bands in runs.values():
  for a in bands.values():a['rmse_prediction_difference']=(a['square']/a['n'])**.5
replay['rmse']=(replay['square']/replay['n'])**.5
out={'scope':'matched frozen weights, 512 fit and 512 dev maps; whole maps with original masks; not official validation','scores':agg,'precision_difference':diff,'b0_replay':replay,'seconds':time.time()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scale_definition':'projection of masked residual onto constant values within fixed s by s blocks, fraction of total SSE; descriptive oracle, not achieved correction'}
(OUT/'frozen_scores.json').write_text(json.dumps(out,indent=2)+'\n');(OUT/'frozen_scores_permap.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in permap))
print(json.dumps({'done':True,'scores':{n:{p:{b:a['rmse'] for b,a in bs.items()} for p,bs in runs.items()} for n,runs in agg.items()},'precision_difference':diff,'replay':replay,'seconds':time.time()-start}),flush=True)
