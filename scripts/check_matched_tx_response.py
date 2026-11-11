"""Matched B0/B1 response to actual same-terrain transmitter changes; no inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'audit/geometry_model_diagnosis_20261031';F=A/'frozen_predictions';z=np.load(A/'development_geometry_samples.npz');fi=json.loads((F/'indices.json').read_text());keep=[j for j,p in enumerate(fi['parts']) if p=='dev'];ids=[fi['index_rows'][j] for j in keep];jm={int(i):j for j,i in enumerate(z['original_rows'])};oldj=[jm[i] for i in ids];rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()));meta=json.loads((ROOT.parent/'data/metadata.json').read_text());wl=np.array([299792458/meta['bands_hz'][b] for b in ('415','58')]);ref=20*np.log10(4*np.pi*z['distance'][oldj,None,:]/wl[None,:,None]);Y=z['labels'][oldj].astype(np.float64)-ref;V=z['valid'][oldj];E=z['E'][oldj];rr=z['receiver_rows'];cc=z['receiver_cols'];pred={}
for name in ['b0_bf16','b1_bf16']:
 p=np.load(F/(name+'.npy'),mmap_mode='r');pred[name]=np.stack([p[j,:,rr,cc].T for j in keep]).astype(np.float64)-ref
by={}
for j,i in enumerate(ids):by.setdefault(rows[i]['terrain_id'],[]).append(j)
result={};pairs=[]
for tid,jj in by.items():
 if len(jj)<2:continue
 xy=np.array([[int(rows[ids[j]]['tx_row']),int(rows[ids[j]]['tx_col'])] for j in jj]);ds=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);np.fill_diagonal(ds,np.inf);target=ds.argmin(1)
 for u,v in sorted({tuple(sorted((k,int(t)))) for k,t in enumerate(target)}):
  j,l=jj[u],jj[v];pairs.append({'terrain_id':tid,'index_rows':[ids[j],ids[l]],'tx_distance':float(ds[u,v])})
  for k,b in enumerate(['415','58']):
   for name,g in [('both_blocked',(E[j]>.25)&(E[l]>.25)),('both_deep_blocked',(E[j]>2)&(E[l]>2))]:
    q=g&V[j,k]&V[l,k];dy=(Y[l,k]-Y[j,k])[q]
    for model,P in pred.items():
     dp=(P[l,k]-P[j,k])[q];e=dp-dy;s=result.setdefault(model,{}).setdefault(b,{}).setdefault(name,{'n':0,'sum_y':0.,'sum_p':0.,'yy':0.,'pp':0.,'yp':0.,'error_square':0.})
     for key,value in {'n':len(dy),'sum_y':float(dy.sum()),'sum_p':float(dp.sum()),'yy':float(dy@dy),'pp':float(dp@dp),'yp':float(dy@dp),'error_square':float(e@e)}.items():s[key]+=value
for bs in result.values():
 for gs in bs.values():
  for s in gs.values():
   n=s['n'];vy=s['yy']/n-(s['sum_y']/n)**2;vp=s['pp']/n-(s['sum_p']/n)**2;cov=s['yp']/n-s['sum_y']*s['sum_p']/n**2;s.update({'label_change_std':vy**.5,'prediction_change_std':vp**.5,'change_correlation':cov/(vy*vp)**.5,'change_error_rmse':(s['error_square']/n)**.5,'descriptive_response_slope':cov/vy})
out={'scope':'same matched512devmaps, nearestTXpairs within identicalterrain, fixedreceivers, normalizedexistinglabels andsavedpredictions; no training or inference','results':result,'pairs':len(pairs),'terrains':len({p['terrain_id'] for p in pairs}),'tx_distance_quantiles':np.quantile([p['tx_distance'] for p in pairs],[.1,.5,.9]).tolist(),'limits':'Descriptive dependent pairs, not independent tests. Does not separate unknown label variation from capacity or optimization causes.'};(A/'matched_tx_change_response.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
