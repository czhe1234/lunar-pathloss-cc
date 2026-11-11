"""Input-only 3D source-direction matching; otherTX labels only diagnostic."""
from pathlib import Path
import json,csv,time,hashlib
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];A=R/'audit/c130_projective'
P=R/'audit/c101_residual/fit';start=time.monotonic()
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
plan=json.loads((R/'audit/c127_joint/protocol.json').read_text())
groups=defaultdict(list)
for i in plan['fit']:groups[rows[i]['terrain_id']].append(i)
groups={k:v for k,v in groups.items() if len(v)==4}
height=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
target=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
masks=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
base=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
bi={int(i):j for j,i in enumerate(json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text()))}
maps=np.load(P/'mapid.npy',mmap_mode='r').reshape(-1,1536)[:,0];pl=np.load(P/'pixel.npy',mmap_mode='r').reshape(-1,1536)
pi={int(i):j for j,i in enumerate(maps)}
rr,cc=np.indices((256,256));rr=rr.ravel();cc=cc.ravel()
# Sufficient statistics per terrain, method, band: n,x,y,x2,y2,xy,difference2,signcorrect,tailn.
records=[];record_ids=[];coverage=[]
names=['same_RX','TX_translation','source_direction']
for terrain,ids in groups.items():
 h=np.asarray(height[int(rows[ids[0]]['hm_row'])],dtype='float64')
 fields=[];trees=[]
 for i in ids:
  r=rows[i];tr=int(r['tx_row']);tc=int(r['tx_col'])
  xyz=np.stack([cc-tc,rr-tr,h.ravel()+1-h[tr,tc]-3],1)
  dist=np.linalg.norm(xyz,axis=1);direction=xyz/dist[:,None]
  y=np.stack([a[i].ravel() for a in target],1).astype('float64')
  valid=np.stack([np.unpackbits(a[i]).reshape(-1) for a in masks],1).astype(bool)
  residual=y-np.asarray(base[bi[i]]).reshape(2,-1).T
  fields.append(dict(tr=tr,tc=tc,direction=direction,distance=dist,res=residual,valid=valid))
  trees.append(cKDTree(direction))
 st=np.zeros((2,3,2,9));cover=np.zeros(4)
 for q,i in enumerate(ids):
  pix=np.asarray(pl[pi[i],:512],dtype='int64');qry=fields[q]
  for d,j in enumerate(ids):
   if q==d:continue
   donor=fields[d];shift_r=rr[pix]+donor['tr']-qry['tr'];shift_c=cc[pix]+donor['tc']-qry['tc']
   inside=(shift_r>=0)&(shift_r<256)&(shift_c>=0)&(shift_c<256)
   shifted=shift_r.clip(0,255)*256+shift_c.clip(0,255)
   angle,match=trees[d].query(qry['direction'][pix],k=1,workers=1)
   angular_miss=angle*np.minimum(qry['distance'][pix],donor['distance'][match])
   idx=[pix,shifted,match];common=qry['valid'][pix]&inside[:,None]
   for at in idx:common&=donor['valid'][at]
   # Exclude negligible source distances identically using geometry, not labels.
   near=(angular_miss<=.5)&(qry['distance'][pix]>=8)&(donor['distance'][match]>=8)
   cover+=np.array([len(pix),inside.sum(),near.sum(),np.mean(angular_miss)])
   for bucket,gate in enumerate([np.ones(len(pix),bool),near]):
    for method,at in enumerate(idx):
     for b in range(2):
      ok=common[:,b]&gate;x=qry['res'][pix,b][ok];y=donor['res'][at,b][ok];tail=np.abs(x)>=5
      st[bucket,method,b]+=np.array([len(x),x.sum(),y.sum(),(x*x).sum(),(y*y).sum(),(x*y).sum(),((x-y)**2).sum(),((x*y>0)&tail).sum(),tail.sum()])
 records.append(st);record_ids.append(int(terrain));coverage.append(cover)
 if len(records)%32==0 or len(records)==len(groups):
  progress=dict(terrains=len(records),total=len(groups),seconds=time.monotonic()-start)
  (A/'progress.json').write_text(json.dumps(progress));print(progress,flush=True)
stats=np.array(records);out=dict(scope='277FITterrains, fourTX each, first512C101querypoints/map, all three methods on identical valid support; no officialVAL, no prediction-input masks',buckets={},seconds=time.monotonic()-start,geometry='publishedH/TX; source+3 receiver+1; no mesh/coordinate reconstruction',interpretation='OtherTX true residuals are privileged diagnostics, not available prediction inputs')
for bk,label in enumerate(['all_common','near_direction_match']):
 z=stats[:,bk].sum(0);out['buckets'][label]={}
 for method,name in enumerate(names):
  s=z[method];n=s[:,0];meanx=s[:,1]/n;meany=s[:,2]/n
  vx=s[:,3]/n-meanx**2;vy=s[:,4]/n-meany**2;cov=s[:,5]/n-meanx*meany
  out['buckets'][label][name]=dict(valid=n.astype(int).tolist(),C82_RMSE=np.sqrt(s[:,3]/n).tolist(),
   direct_label_transfer_RMSE=np.sqrt(s[:,6]/n).tolist(),residual_correlation=(cov/np.sqrt(vx*vy)).tolist(),
   tail_direction=(s[:,7]/np.maximum(s[:,8],1)).tolist(),tail_count=s[:,8].astype(int).tolist())
out['status']='complete';out['train_launched']=False
np.savez(A/'projective_stats.npz',stats=stats,terrain_ids=record_ids,coverage=coverage,methods=names)
(A/'projection_result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
