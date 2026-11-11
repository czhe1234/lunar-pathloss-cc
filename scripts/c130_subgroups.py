"""Fixed input-defined partitions; diagnostic only, no selection or new fitted gates."""
from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c130_projective';out={}
for part in ['fit','monitor']:
 P=R/'audit/c101_residual'/part;rows=np.load(A/'nodes'/part/'rows.npy');meta=np.load(A/'nodes'/part/'meta.npy',mmap_mode='r')
 target=np.load(P/'target.npy',mmap_mode='r')[rows];base=np.load(P/'base.npy',mmap_mode='r')[rows];r=target[:,:2].astype('float64')-base
 valid=(target[:,2:]>0)&(np.abs(r)>=5)
 geometric_positive=meta[:,0,1]>0
 metadata=list(csv.DictReader((R.parent/'data/train_index.csv').open()));mi=np.load(P/'mapid.npy',mmap_mode='r')[rows];px=np.load(P/'pixel.npy',mmap_mode='r')[rows]
 tr=np.array([int(metadata[int(i)]['tx_row']) for i in mi]);tc=np.array([int(metadata[int(i)]['tx_col']) for i in mi]);span=np.maximum(np.abs(px//256-tr),np.abs(px%256-tc))
 remote=(meta[:,0,0]*span>8)&((1-meta[:,0,0])*span>16)
 groups=dict(primary_obstructed=geometric_positive,primary_nonobstructed=~geometric_positive,primary_outside_old_endpoint_patches=remote,primary_near_endpoint=~remote)
 models={m:np.load(R/f'runs/c130_nodes_{m}/{part}_delta.npy').astype('float64') for m in ['old','nodes','shuffled']}
 out[part]={}
 for group,flag in groups.items():
  mask=valid&flag[:,None];count=mask.sum(0)
  scores={m:np.sqrt(np.where(mask,(d-r)**2,0).sum(0)/count).tolist() for m,d in models.items()}
  out[part][group]=dict(tail_count=count.tolist(),tail_fraction=(count/valid.sum(0)).tolist(),tail_RMSE=scores,
   gain_nodes_vs_old=(np.array(scores['old'])-scores['nodes']).tolist())
out['scope']='post-hoc fixed geometry partitions; primary raw clearance>0 and center outside the original axis-aligned17x17TX and33x33RXpatches; overlapping partitions, not additive; no thresholds tuned, no training or deployment gate derived'
(A/'subgroups.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
