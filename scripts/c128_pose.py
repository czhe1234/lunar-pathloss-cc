import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';A=R/'audit/c128_pose'
p=json.loads((R/'audit/c127_joint/protocol.json').read_text());rows=list(csv.DictReader(open(D/'train_index.csv')))
ya=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ma=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
ix=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());lookup={int(i):j for j,i in enumerate(ix)}
base=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
allstats=[];terrains=[]
for terrain,positions in p['groups'].items():
 if len(positions)!=4:continue
 ids=[p['fit'][j] for j in positions];b=np.asarray(base[[lookup[i] for i in ids]],np.float64)
 y=np.stack([v[ids] for v in ya],1).astype(np.float64);v=np.stack([np.unpackbits(a[ids],axis=1).reshape(4,256,256).astype(bool) for a in ma],1)
 residual=np.where(v,y-b,0.)
 stat=np.zeros((2,10),np.float64)
 for j in range(4):
  original=np.zeros((2,256,256));shifted=original.copy();num=original.copy()
  for k in range(4):
   if k==j:continue
   dr=int(rows[ids[k]]['tx_row'])-int(rows[ids[j]]['tx_row']);dc=int(rows[ids[k]]['tx_col'])-int(rows[ids[j]]['tx_col'])
   rr0=max(0,-dr);rr1=min(256,256-dr);cc0=max(0,-dc);cc1=min(256,256-dc)
   dst=(slice(None),slice(rr0,rr1),slice(cc0,cc1));src=(slice(None),slice(rr0+dr,rr1+dr),slice(cc0+dc,cc1+dc))
   ok=v[k][dst]&v[k][src]
   original[dst]+=np.where(ok,residual[k][dst],0);shifted[dst]+=np.where(ok,residual[k][src],0);num[dst]+=ok
  common=v[j]&(num>0);original/=np.maximum(num,1);shifted/=np.maximum(num,1)
  r=np.where(common,residual[j],0);tail=common&(np.abs(r)>=5)
  stat[:,0]+=(r*r).sum((1,2));stat[:,1]+=np.where(common,(r-original)**2,0).sum((1,2));stat[:,2]+=np.where(common,(r-shifted)**2,0).sum((1,2));stat[:,3]+=common.sum((1,2))
  stat[:,4]+=(tail&(original*r>0)).sum((1,2));stat[:,5]+=(tail&(shifted*r>0)).sum((1,2));stat[:,6]+=tail.sum((1,2))
  stat[:,7]+=np.where(common,r*shifted,0).sum((1,2));stat[:,8]+=np.where(common,shifted*shifted,0).sum((1,2));stat[:,9]+=v[j].sum((1,2))
 allstats.append(stat);terrains.append(terrain)
a=np.stack(allstats);s=a.sum(0)
rng=np.random.default_rng(128073);draw=rng.integers(len(a),size=(3000,len(a)));z=a[draw].sum(1);rm=np.sqrt(z[:,:,:3]/z[:,:,3,None])
out=dict(scope='277FITterrains/1108maps; matched donor availability and query pixels; oracle diagnosis with otherTXlabels,not deployable',
 reference=np.sqrt(s[:,0]/s[:,3]).tolist(),same_RX_donor=np.sqrt(s[:,1]/s[:,3]).tolist(),TX_relative_donor=np.sqrt(s[:,2]/s[:,3]).tolist(),
 relative_gain_vs_reference=(np.sqrt(s[:,0]/s[:,3])-np.sqrt(s[:,2]/s[:,3])).tolist(),
 relative_gain_CI95=np.quantile(rm[:,:,0]-rm[:,:,2],[.025,.975],axis=0).T.tolist(),
 direction_same_RX=(s[:,4]/s[:,6]).tolist(),direction_TX_relative=(s[:,5]/s[:,6]).tolist(),
 valid=s[:,3].astype(int).tolist(),coverage=(s[:,3]/s[:,9]).tolist(),cross_moment=(s[:,7]/s[:,3]).tolist(),
 official_val_used=False,train_launched=False)
np.savez(A/'terrain_stats.npz',stats=a,terrain=np.array(terrains))
(A/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
