from pathlib import Path
import csv,json,time
import numpy as np
R=Path(__file__).resolve().parents[1]
A=R/'audit/c116_targeted';P=R/'audit/c101_residual'
def load(p): return np.load(p,mmap_mode='r')
def report(s,c):return np.sqrt(s/np.maximum(c,1)).tolist()
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
seen72=set(json.loads((R/'runs/c72_observed/protocol.json').read_text())['fit'])
seen73=set(json.loads((R/'runs/c73_wide_real/protocol.json').read_text())['fit'])
result={'official_val_used':False,'scope':'original128monitor terrains, same2048points permap; local oracles use true labels and are NOT deployable'}
groups={}
for part in ['fit','monitor']:
 b=load(P/part/'base.npy');y=load(P/part/'target.npy');mi=load(P/part/'mapid.npy')
 for name,sel in [('all',np.ones(len(mi),bool)),('C72map_seen',np.isin(mi,list(seen72))),('C72map_unseen',~np.isin(mi,list(seen72))),('C73map_seen',np.isin(mi,list(seen73))),('C73map_unseen',~np.isin(mi,list(seen73)))]:
  e=y[sel,:2].astype('float64')-b[sel];m=y[sel,2:4].astype(bool)
  if not m.any():continue
  s=(e*e*m).sum(0);c=m.sum(0);tail=m&(e>=5)
  groups[part+'_'+name]={'rmse':report(s,c),'count':c.tolist(),'positive5_fraction':(tail.sum(0)/c).tolist(),'positive5_SSE_fraction':((e*e*tail).sum(0)/s).tolist(),'mean':(np.where(m,e,0).sum(0)/c).tolist()}
result['exposure']=groups
ids=json.loads((P/'monitor/done.json').read_text())['ids']
bindex={i:j for j,i in enumerate(json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'])}
field=load(R/'audit/c88_mesh/base.npy');pixels=load(P/'monitor/pixel.npy').reshape(len(ids),-1)
target=load(P/'monitor/target.npy').reshape(len(ids),-1,4)
base=load(P/'monitor/base.npy').reshape(len(ids),-1,2)
stats={}
for radius in [1,2,4]:
 offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
 s=np.zeros((3,2));c=np.zeros(2);weakc=np.zeros(2);weaks=np.zeros((3,2))
 outside=np.zeros((2,2));hist=np.zeros(len(offsets),np.int64)
 for j,i in enumerate(ids):
  yy=target[j,:,:2].astype('float64');m=target[j,:,2:4].astype(bool);b=base[j].astype('float64')
  rr=pixels[j]//256;cc=pixels[j]%256
  dr=np.array([z[0] for z in offsets]);dc=np.array([z[1] for z in offsets])
  ri=(rr[None]+dr[:,None]).clip(0,255);ci=(cc[None]+dc[:,None]).clip(0,255)
  bank=field[bindex[i]][:,ri,ci].transpose(1,2,0).astype('float64')
  assert np.array_equal(bank[offsets.index((0,0))],b)
  err=(bank-yy[None])**2
  common=(err*m[None]).sum(-1).argmin(0)
  joint=bank[common,np.arange(len(rr))]
  independent=np.take_along_axis(bank,err.argmin(0)[None],axis=0)[0]
  weak=m&((yy-b)>=5)
  for k,pred in enumerate([b,joint,independent]):
   e=(pred-yy)**2;s[k]+=(e*m).sum(0);weaks[k]+=(e*weak).sum(0)
  c+=m.sum(0);weakc+=weak.sum(0)
  maximum=bank.max(0)
  for k,margin in enumerate([1,5]):outside[k]+=(weak&(yy>maximum+margin)).sum(0)
  hist+=np.bincount(common[m.any(1)],minlength=len(offsets))
 stats[str(radius)]={'baseline':report(s[0],c),'common_shift_ORACLE':report(s[1],c),'independent_shift_ORACLE':report(s[2],c),
  'weak_RMSE':[report(z,weakc) for z in weaks],'weak_count':weakc.tolist(),'weak_outside_bank_plus1_plus5_fraction':(outside/weakc).tolist(),
  'count':c.tolist(),'common_shift_identity_fraction':float(hist[offsets.index((0,0))]/hist.sum())}
 print(radius,stats[str(radius)],flush=True)
result['spatial_bank']=stats
(A/'diagnosis.json').write_text(json.dumps(result,indent=2));print('DONE',flush=True)
