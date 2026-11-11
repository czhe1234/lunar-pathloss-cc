"""Privileged same-band observation control, no training.
Fixed query grid and supports at four radii, no query label enters prediction.
All radii compared on identical common-valid queries. Not deployable.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,time
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'audit/c18_identifiability_20261101'
C=R/'features/c17_spatial';meta=json.loads((C/'metadata.json').read_text());ids=meta['ids']['dev']
rows=list(csv.DictReader((D/'train_index.csv').open()))
X=np.load(C/'dev_x.npy',mmap_mode='r');P=np.load(C/'dev_base.npy',mmap_mode='r')
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
rr,cc=np.meshgrid(np.arange(8,248,16),np.arange(8,248,16),indexing='ij');rr=rr.ravel();cc=cc.ravel()
radii=[1,2,4,8];arrays={n:[] for n in ['C10']+[f'{mode}_radius{d}' for d in radii for mode in ['residual_interpolation','label_interpolation']]}
truth=[];valid=[];zones=[];rawvalid=[]
for j,i in enumerate(ids):
 yy=np.stack([v[i] for v in Y]).astype(float);mm=np.stack([np.unpackbits(v[i],bitorder='big').reshape(256,256) for v in M]).astype(bool);pp=np.array(P[j],float)
 yq=yy[:,rr,cc].T;pq=pp[:,rr,cc].T;mv=mm[:,rr,cc].T.copy();rawvalid.append(mv.copy());truth.append(yq);arrays['C10'].append(pq)
 f=np.array(X[j,:31],np.float32);e3=(2*np.sinh(3*f[11:19])).max(0);e2=2*np.sinh(3*f[19]);strict=(e3[rr,cc]>=0)
 z={'all':np.ones(len(rr),bool),'occluded_proxy':e2[rr,cc]>=0,'height_sensitive':(e3[rr,cc]<0)&(e2[rr,cc]>=0),'nominal_blocked':e3[rr,cc]>=0}
 for d in radii:
  r=np.stack((rr-d,rr-d,rr+d,rr+d));c=np.stack((cc-d,cc+d,cc-d,cc+d))
  sy=yy[:,r,c];sp=pp[:,r,c];sm=mm[:,r,c]
  mv&=sm.all(1).T;strict&=(e3[r,c]>=0).all(0)
  arrays[f'label_interpolation_radius{d}'].append(sy.mean(1).T)
  arrays[f'residual_interpolation_radius{d}'].append(pq+(sy-sp).mean(1).T)
 z['blocked_at_query_and_all_supports']=strict;zones.append(z);valid.append(mv)
T=np.stack(truth);V=np.stack(valid);RV=np.stack(rawvalid);ZN={z:np.stack([v[z] for v in zones]) for z in zones[0]}
rng=np.random.default_rng(20261106);w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=1000)
stats={};ssby={}
for name,values in arrays.items():
 a=np.stack(values);stats[name]={};ssby[name]={}
 for z,zmask in ZN.items():
  stats[name][z]={};ssby[name][z]={}
  for k,b in enumerate(['415','58']):
   m=V[:,:,k]&zmask;ss=(((a[:,:,k]-T[:,:,k])**2)*m).sum(1);count=m.sum(1);ssby[name][z][b]=ss;basess=ssby['C10'][z][b] if name!='C10' else ss
   stats[name][z][b]={'common_valid_pixels':int(count.sum()),'valid_query_before_support_requirement':int((RV[:,:,k]&zmask).sum()),'RMSE':float(np.sqrt(ss.sum()/count.sum())) if count.sum() else None,
   'SSE_reduction_percent':float(100*(1-ss.sum()/basess.sum())) if basess.sum() else None,
   'terrain_bootstrap95':np.percentile(100*(1-(w@ss)/np.maximum(w@basess,1e-15)),[2.5,97.5]).tolist()}
out={'state':'completed','scope':'128 distinct internalDEV terrains from the existing C17 input-only selection; one TX per terrain; no model training',
 'query_rule':'Rows/columns8,24,...,232,225 queries/map. All support pixels disjoint from queries.',
 'support_rule':'Four diagonal support pixels at +/-d row and col, d=1,2,4,8 metres. Mean of four support residuals or labels. No query label enters prediction.',
 'same_evaluation_pixels':'For all radii, only queries valid in the given band and all16 supports are scored. This mask restriction can select easier regions; counts retained.',
 'nominal_blocked_control':'blocked_at_query_and_all_supports requires nominalTX3 clearance>=0 at query and every support point, independent of labels.',
 'statistics':stats,'maps_per_band':len(ids),'query_points':len(ids)*len(rr),'index_rows':ids,
 'limits':'Neighbour labels are additional privileged observations unavailable at deployment. Spatial predictability does not prove predictability from H+TX; interpolation may also smooth spatially correlated simulator noise.'}
(O/'spatial_observation_control.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'state':'completed','statistics':{n:{z:v[z] for z in ['occluded_proxy','blocked_at_query_and_all_supports']} for n,v in stats.items() if n=='C10' or n.startswith('residual_interpolation')}},indent=2))
