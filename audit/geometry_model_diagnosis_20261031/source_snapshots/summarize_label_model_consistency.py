"""Descriptive checks of saved labels and predictions; no fitted parameters."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/geometry_model_diagnosis_20261031';DATA=ROOT.parent/'data'
a=np.load(OUT/'development_geometry_samples.npz');Y=a['labels'].astype(np.float64);P=a['predictions'].astype(np.float64);E=a['E'];V=a['valid'];distance=a['distance'];cosine=a['cosine']
meta=json.loads((DATA/'metadata.json').read_text());wavelength=np.array([299792458./meta['bands_hz'][b] for b in ('415','58')]);reference=20*np.log10(4*np.pi*distance[:,None,:]/wavelength[None,:,None]);YL=Y-reference;PL=P-reference;err=P-Y
zones={'all':np.ones_like(E,bool),'strong_clear':E<-.5,'very_strong_clear':E<-.75,'clear':E<-.25,'critical':np.abs(E)<=.25,'blocked':E>.25}
cosbins=[0,.001,.003,.01,.03,.1,1.0001]
results={};examples=[]
for k,b in enumerate(('415','58')):
 total_sse=float((err[:,k][V[:,k]]**2).sum());total_n=int(V[:,k].sum());r={}
 def stat(q):
  q=q&V[:,k];e=err[:,k][q];y=YL[:,k][q];p=PL[:,k][q];n=len(e)
  return {'n':n,'pixel_share':n/total_n,'sse_share':float((e*e).sum())/total_sse,'rmse':float(np.sqrt((e*e).mean())) if n else None,'mean_error':float(e.mean()) if n else None,'label_excess_mean':float(y.mean()) if n else None,'label_excess_quantiles':{str(v):float(np.quantile(y,v)) for v in (.01,.1,.5,.9,.99)} if n else {},'prediction_excess_mean':float(p.mean()) if n else None}
 for name,z in zones.items():
  r[name]=stat(z)
  for limit in (3,6,10):r[name+f'_label_excess_gt{limit}']=stat(z&(YL[:,k]>limit))
 for low,high in zip(cosbins[:-1],cosbins[1:]):r[f'strong_clear_cos_{low}_{high}']=stat((E<-.5)&(cosine>=low)&(cosine<high))
 states_y=np.digitize(YL[:,k],[3.,10.]);states_p=np.digitize(PL[:,k],[3.,10.]);matrix={}
 for yi in range(3):
  for pi in range(3):matrix[f'label{yi}_prediction{pi}']=stat((states_y==yi)&(states_p==pi))
 results[b]={'groups':r,'state_confusion':matrix,'state_definition':'0 excess<=3 dB, 1 between3and10, 2 excess>=10 dB; proxy power states, not true physical path labels'}
# Discovery examples with both bands anomalously attenuated in strongly clear geometry.
selection=json.loads((OUT/'geometry_selection.json').read_text())['items'];disc={x['index_row'] for x in selection if x['cohort']=='discovery'};map_disc=np.array([int(i) in disc for i in a['original_rows']])
q=(E<-.5)&V.all(1)&(YL.min(1)>10)&map_disc[:,None]
loc=np.argwhere(q)
if len(loc):
 strength=YL.min(1)[q];order=np.argsort(strength);chosen=[order[int(v*(len(order)-1))] for v in (.25,.5,.75,.95)]
 for index in chosen:
  j,s=loc[index];examples.append({'prediction_row':int(j),'original_row':int(a['original_rows'][j]),'receiver_row':int(a['receiver_rows'][s]),'receiver_col':int(a['receiver_cols'][s]),'E':float(E[j,s]),'label_excess_db':YL[j,:,s].tolist(),'model_error_db':err[j,:,s].tolist(),'cosine':float(cosine[j,s])})
out={'scope':'2562 development maps, fixed1024sampledreceivers; original TX3/RX1 geometry; normalized power is a diagnostic transformation of released labels','bands':results,'examples':examples}
(OUT/'label_model_consistency.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'strong_clear':{b:{n:r['groups'][n] for n in ['strong_clear','strong_clear_label_excess_gt6','strong_clear_label_excess_gt10']} for b,r in results.items()},'examples':examples}),flush=True)
