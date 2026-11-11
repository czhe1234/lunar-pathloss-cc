"""Original-mask CPU score of frozen symmetry outputs; no parameter fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'audit/geometry_model_diagnosis_20261031';F=A/'symmetry_predictions';D=ROOT.parent/'data';assert (F/'inference.json').exists();idx=json.loads((F/'indices.json').read_text());table=list(csv.DictReader((D/'train_index.csv').open()));p={n:np.load(F/(n+'.npy'),mmap_mode='r') for n in ('canonical','mean8','variance8')};old=np.load(A/'frozen_predictions/b1_bf16.npy',mmap_mode='r');Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')];total={};terrain={};permap=[];replay={'max_abs':0.,'square':0.,'n':0};start=time.time()
for j,(i,part) in enumerate(zip(idx['index_rows'],idx['parts'])):
 tid=table[i]['terrain_id']
 for k,b in enumerate(('415','58')):
  v=np.unpackbits(M[k][i],bitorder='big').reshape(256,256).astype(bool);y=Y[k][i].astype(np.float64);e=p['canonical'][j,k].astype(np.float64)-y;em=p['mean8'][j,k].astype(np.float64)-y;var=p['variance8'][j,k].astype(np.float64);s={'n':int(v.sum()),'canonical_sse':float((e[v]**2).sum()),'ensemble_sse':float((em[v]**2).sum()),'orientation_variance':float(var[v].sum()),'old_sse':float(((old[j,k].astype(np.float64)-y)[v]**2).sum())}
  for a in [total.setdefault(part,{}).setdefault(b,{}),terrain.setdefault(part,{}).setdefault(b,{}).setdefault(tid,{})]:
   for key,value in s.items():a[key]=a.get(key,0)+value
  d=p['canonical'][j,k].astype(np.float64)-old[j,k].astype(np.float64);replay['max_abs']=max(replay['max_abs'],float(abs(d).max()));replay['square']+=float((d*d).sum());replay['n']+=d.size
  permap.append({'index_row':i,'terrain_id':tid,'part':part,'band':b,**s})
for part,bs in total.items():
 for b,s in bs.items():
  s['canonical_rmse']=(s['canonical_sse']/s['n'])**.5;s['ensemble_rmse']=(s['ensemble_sse']/s['n'])**.5;s['old_rmse']=(s['old_sse']/s['n'])**.5;s['orientation_std_rms']=(s['orientation_variance']/s['n'])**.5;s['mean_orientation_rmse']=((s['ensemble_sse']+s['orientation_variance'])/s['n'])**.5;s['gain_db']=s['canonical_rmse']-s['ensemble_rmse'];s['orientation_variance_share_of_mean_mse']=s['orientation_variance']/(s['ensemble_sse']+s['orientation_variance'])
  arr=np.array([[q['n'],q['canonical_sse'],q['ensemble_sse']] for q in terrain[part][b].values()]);rng=np.random.default_rng(2026103106);draw=rng.integers(0,len(arr),size=(2000,len(arr)));sums=arr[draw].sum(1);g=np.sqrt(sums[:,1]/sums[:,0])-np.sqrt(sums[:,2]/sums[:,0]);s['terrain_bootstrap_gain_95pct']=np.quantile(g,[.025,.975]).tolist();s['terrain_count']=len(arr)
replay['rmse']=(replay['square']/replay['n'])**.5
out={'scope':'512fit and512dev maps, same original masks, no training; not official validation','scores':total,'replay_against_previous_frozen_b1':replay,'seconds':time.time()-start};(A/'symmetry_scores.json').write_text(json.dumps(out,indent=2)+'\n');(A/'symmetry_permap.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in permap));print(json.dumps({'done':True,**out}),flush=True)
