"""Natural same-terrain TX changes using existing labels and B0 predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'audit/geometry_model_diagnosis_20261031';z=np.load(A/'development_geometry_samples.npz');ids=z['original_rows'];rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()));Y=z['labels'].astype(np.float64);P=z['predictions'].astype(np.float64);V=z['valid'];E=z['E'];meta=json.loads((ROOT.parent/'data/metadata.json').read_text());wl=np.array([299792458/meta['bands_hz'][b] for b in ('415','58')]);ref=20*np.log10(4*np.pi*z['distance'][:,None,:]/wl[None,:,None]);Y-=ref;P-=ref;by={}
for j,i in enumerate(ids):by.setdefault(rows[int(i)]['terrain_id'],[]).append(j)
result={};pairs=[]
for tid,jj in by.items():
 if len(jj)<2:continue
 xy=np.array([[int(rows[int(ids[j])]['tx_row']),int(rows[int(ids[j])]['tx_col'])] for j in jj]);ds=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
 for kind in ('nearest','farthest'):
  dd=ds.copy();np.fill_diagonal(dd,np.inf if kind=='nearest' else -1);target=dd.argmin(1) if kind=='nearest' else dd.argmax(1);pp={tuple(sorted((k,int(t)))) for k,t in enumerate(target)}
  for u,v in sorted(pp):
   j=jj[u];l=jj[v];dist=float(ds[u,v]);pairs.append({'terrain_id':tid,'kind':kind,'index_rows':[int(ids[j]),int(ids[l])],'tx_distance':dist})
   gates={'both_clear':(E[j]<-.25)&(E[l]<-.25),'both_blocked':(E[j]>.25)&(E[l]>.25),'both_deep_blocked':(E[j]>2)&(E[l]>2),'state_switch':((E[j]<-.25)&(E[l]>.25))|((E[l]<-.25)&(E[j]>.25))}
   for k,b in enumerate(('415','58')):
    for name,g in gates.items():
     q=g&V[j,k]&V[l,k];dy=(Y[l,k]-Y[j,k])[q];dp=(P[l,k]-P[j,k])[q];e=dp-dy;s=result.setdefault(kind,{}).setdefault(b,{}).setdefault(name,{'n':0,'sum_y':0.,'sum_p':0.,'yy':0.,'pp':0.,'yp':0.,'error_square':0.})
     for key,value in {'n':len(dy),'sum_y':float(dy.sum()),'sum_p':float(dp.sum()),'yy':float(dy@dy),'pp':float(dp@dp),'yp':float(dy@dp),'error_square':float(e@e)}.items():s[key]+=value
for bs in result.values():
 for gs in bs.values():
  for s in gs.values():
   n=s['n'];vy=s['yy']/n-(s['sum_y']/n)**2;vp=s['pp']/n-(s['sum_p']/n)**2;cov=s['yp']/n-s['sum_y']*s['sum_p']/n**2;s.update({'label_change_std':vy**.5,'prediction_change_std':vp**.5,'change_correlation':cov/(vy*vp)**.5,'change_error_rmse':(s['error_square']/n)**.5,'descriptive_response_slope':cov/vy})
out={'scope':'39 dev terrains with51TX; unique nearest andfarthestTXpairs within eachterrain; fixed1024receiverpositions; normalizedexistinglabels andpredictions; no fit, no new simulation','results':result,'pair_counts':{k:sum(p['kind']==k for p in pairs) for k in ['nearest','farthest']},'tx_distance_quantiles':{k:np.quantile([p['tx_distance'] for p in pairs if p['kind']==k],[.1,.5,.9]).tolist() for k in ['nearest','farthest']},'limits':'Pairs share maps, pixels and terrain and are not independent. Descriptive response slope/correlation cannot separate unpredictable label variation from representation/optimization limits.'};(A/'tx_change_response.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
