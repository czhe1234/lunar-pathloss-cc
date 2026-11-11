import sys,json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
R=Path(__file__).resolve().parents[1];A=R/'audit/c121_pairs';B=R/'audit/c122_power'
r=json.load(open(A/'monitor/pairs.json'));f=np.load(A/'monitor/pair_features.npz');old=f['old'];meta=f['metadata'];proto=json.load(open(A/'power_probe_protocol.json'));maps=set(proto['maps'])
keep=np.array([i for i,x in enumerate(r) if x['mapid'] in maps]);bands=np.array([r[i]['band'] for i in keep]);lookup=np.load(R/'src/c70_centroid_lookup0.npz');ids=lookup['ids'];w=lookup['weights']
pred={}
for mode,dirs,floor in [('clip128',128,1e-12),('wide128',128,1e-38),('wide8192',8192,1e-38)]:
 z=[]
 for i in keep:
  e=r[i];px=np.array([e['anchor_pixel'],e['opposite_pixel'],e['same_pixel'] if e['same_pixel']>=0 else e['anchor_pixel']])
  order=np.argsort(-w[px],axis=1,kind='stable');pick=np.take_along_axis(ids[px],order,1);weight=np.take_along_axis(w[px],order,1)
  raw=np.load(A/f'power{dirs}/{e["mapid"]}.npy')[pick].astype('float64').clip(floor,1e8);log=np.log10(raw)/4
  feats=np.c_[log.reshape(3,30),(log*weight[:,:,None]).sum(1),np.log10((raw*weight[:,:,None]).sum(1))/4,meta[i]]
  z.append(np.c_[old[i],feats])
 z=np.array(z);p=np.zeros(z.shape[:2])
 for b in [0,1]:
  m=CatBoostClassifier();m.load_model(str(R/f'runs/c122_{mode}/band{b}.cbm'));select=bands==b;p[select]=m.predict_proba(z[select].reshape(-1,228),thread_count=4)[:,1].reshape(-1,3)
 pred[mode]=p
out={'scope':'paired subset of24 power-probe maps; conditional true-tail prediction only','n':len(keep),'modes':{},'examples':[]}
for mode,p in pred.items():
 out['modes'][mode]=[]
 for b in range(2):
  q=p[bands==b];out['modes'][mode].append(dict(band=b,n=len(q),both_directions_correct=float(((q[:,0]>=.5)&(q[:,1]<.5)).mean()),anchor_probability_higher=float((q[:,0]>q[:,1]).mean())))
for ex in json.load(open(A/'pair_report.json'))['examples']:
 ix=np.flatnonzero(keep==ex['row_index'])[0];out['examples'].append(dict(**ex,conditional_sign_probability={m:pred[m][ix].tolist() for m in pred}))
(B/'pair_evaluation.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out['modes'],indent=2))
