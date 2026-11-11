"""CPU raw-label FP64 regional and tail diagnostics; no training."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
ids=json.loads((R/'audit/c132_transport/protocol.json').read_text())['monitor']
bi={i:j for j,i in enumerate(json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'])}
base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
names=['c132_direction','c133_direction','c133_local'];pred={n:np.load(R/'runs'/n/'full_monitor/prediction.npy',mmap_mode='r') for n in names}
y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];v=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
vis=np.load(R/'audit/c131_transport/direct_n16_monitor/power.npy',mmap_mode='r')
vi={i:j for j,i in enumerate(json.loads((R/'audit/c131_transport/protocol.json').read_text())['monitor'])}
stats={n:{r:np.zeros((2,8),np.float64) for r in ['all','tail','ordinary','full_visible','partial','blocked']} for n in names}
for j,i in enumerate(ids):
 b=np.array(base[bi[i]],dtype='float64');yy=np.stack([a[i] for a in y]).astype('float64');valid=np.stack([np.unpackbits(a[i]).reshape(256,256) for a in v]).astype(bool)
 need=yy-b;vv=np.array(vis[vi[i],3]);tail=abs(need)>=5
 masks=dict(all=valid,tail=valid&tail,ordinary=valid&~tail,full_visible=valid&(vv>.999),partial=valid&(vv>0)&(vv<=.999),blocked=valid&(vv==0))
 for name in names:
  pp=np.array(pred[name][j],dtype='float64');c=pp-b;err=pp-yy
  for k,m in masks.items():
   vals=[np.where(m,z,0).sum((1,2)) for z in [err**2,need**2,np.ones_like(need),abs(need),abs(c),c*need,c**2,(c*need>0)]]
   stats[name][k]+=np.stack(vals,1)
out={}
for n,regions in stats.items():
 out[n]={}
 for r,z in regions.items():
  out[n][r]=dict(RMSE=np.sqrt(z[:,0]/z[:,2]).tolist(),C82=np.sqrt(z[:,1]/z[:,2]).tolist(),count=z[:,2].tolist(),
   needed_abs_db=(z[:,3]/z[:,2]).tolist(),correction_abs_db=(z[:,4]/z[:,2]).tolist(),
   correction_residual_cosine=(z[:,5]/np.sqrt(np.maximum(z[:,1]*z[:,6],1e-40))).tolist(),direction_accuracy=(z[:,7]/z[:,2]).tolist())
result=dict(status='complete',scope='all628monitor maps; raw TRAIN labels, masks, FP64; full-visible is candidate geometry, not official state',results=out)
(R/'audit/c134_paired/previous_branch_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({n:{r:d[r] for r in ['all','tail']} for n,d in out.items()},indent=2))
