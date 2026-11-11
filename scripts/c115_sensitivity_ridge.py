"""Terrain-heldout sufficient-statistic ridge probe for H/TX-derived sensitivity information."""
import json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path.cwd();D=R.parent/'data';A=R/'audit/c115_TXsensitivity'
plan=json.loads((R/'audit/c112_TXresponse/protocol.json').read_text())
pred={}
for shard in [0,1]:
 p=R/f'audit/c112_TXresponse/shard{shard}';ids=json.loads((p/'ids.json').read_text());array=np.load(p/'prediction.npy',mmap_mode='r')
 pred.update({i:array[j] for j,i in enumerate(ids)})
rows=list(csv.DictReader(open(D/'train_index.csv')));hm=np.load(D/'train_hm.npy',mmap_mode='r')
yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ww=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
gy,gx=np.mgrid[:256,:256];stats=defaultdict(lambda:[None,None]);dims={}
for pair in plan['pairs']:
 if pair['category']!='unit1m':continue
 ids=pair['ids'];terrain=pair['terrain'];h=np.array(hm[int(rows[ids[0]]['hm_row'])],np.float64);sy,sx=np.gradient(h)
 rr=[int(rows[i]['tx_row']) for i in ids];cc=[int(rows[i]['tx_col']) for i in ids]
 remote=np.ones((256,256),bool)
 for r,c in zip(rr,cc):remote&=((gx-c)**2+(gy-r)**2>=100)
 for side in [0,1]:
  i,j=ids[side],ids[1-side];tr=rr[side];tc=cc[side];dr=rr[1-side]-tr;dc=cc[1-side]-tc
  p=np.asarray(pred[i],np.float64);q=np.asarray(pred[j],np.float64);diff=q-p;dx=(gx-tc)/128;dy=(gy-tr)/128
  dz=(h+1-(float(h[tr,tc])+3))/50;distance=np.sqrt((dx*128)**2+(dy*128)**2+(dz*50)**2).clip(.25)
  grads=np.array([np.gradient(p[b],axis=a) for b in [0,1] for a in [0,1]])
  base=np.stack([np.ones_like(h),p[0]/100,p[1]/100,h/100,sx/5,sy/5,dx,dy,dz,np.log1p(distance)/6,*[g/10 for g in grads]],-1).reshape(-1,14)
  extra=np.stack([diff[0]/10,diff[1]/10,abs(diff[0])/10,abs(diff[1])/10,
   diff[0]*dr/10,diff[0]*dc/10,diff[1]*dr/10,diff[1]*dc/10,diff[0]**2/100,diff[1]**2/100,diff[0]*diff[1]/100],-1).reshape(-1,11)
  x=np.concatenate([base,extra],1);dims={'baseline':14,'sensitivity':25}
  for b in [0,1]:
   mask=remote&np.unpackbits(ww[b][i]).reshape(256,256).astype(bool)&np.unpackbits(ww[b][j]).reshape(256,256).astype(bool)
   X=x[mask.reshape(-1)];target=np.asarray(yy[b][i],np.float64)[mask]-p[b][mask];weak=target>=5
   G=X.T@X;g=X.T@target;v=float(target@target);GG=X[weak].T@X[weak];gg=X[weak].T@target[weak];vv=float(target[weak]@target[weak])
   row=[G,g,v,len(target),GG,gg,vv,int(weak.sum())]
   if stats[terrain][b] is None:stats[terrain][b]=row
   else:
    for k in range(8):stats[terrain][b][k]+=row[k]
keys=sorted(stats);out={}
for b in [0,1]:
 sums=[sum((stats[k][b][j] for k in keys)) for j in range(8)]
 for name,d in dims.items():
  SSE=0.;weakSSE=0.;trainSSE=0.
  for k in keys:
   G,g,v,n,GG,gg,vv,nw=stats[k][b]
   trainG=sums[0][:d,:d]-G[:d,:d];traing=sums[1][:d]-g[:d];trainN=sums[3]-n
   ridge=1e-3*trainN*np.eye(d);ridge[0,0]=0
   beta=np.linalg.solve(trainG+ridge,traing)
   SSE+=v-2*beta@g[:d]+beta@G[:d,:d]@beta
   weakSSE+=vv-2*beta@gg[:d]+beta@GG[:d,:d]@beta
  out.setdefault(name,[]).append(dict(count=int(sums[3]),rmse=float(np.sqrt(max(SSE,0)/sums[3])),
   weak_count=int(sums[7]),weak_power_rmse=float(np.sqrt(max(weakSSE,0)/sums[7]))))
 out.setdefault('C82',[]).append(dict(count=int(sums[3]),rmse=float(np.sqrt(sums[2]/sums[3])),weak_count=int(sums[7]),weak_power_rmse=float(np.sqrt(sums[6]/sums[7]))))
result=dict(terrains=len(keys),methods=out,source='fixed37FITterrains,allrecordsfromsameterrainleftouttogether;predictionsinputonly,noquerylabelsinputs',
 ridge_lambda=1e-3,features={'baseline':'14H/TX+C82valueandRXgradient','sensitivity':'baseline+11finiteTXdifferencesandinteractions'},
 target='Y-C82 at each endpoint; commonvalidreceiver>=10m;notoriginal484holdout,notpromotionresult',official_VAL_used=False)
(A/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
