"""Original raw TRAIN labels and masks; independent conditional scoring."""
from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c130_projective';S=R/'audit/c101_residual';D=R.parent/'data'
ys=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ms=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
tm=np.array([int(r['terrain_id']) for r in csv.DictReader((D/'train_index.csv').open())])
results={};monitor_stats={}
for part in ['fit','monitor']:
 P=S/part;mi=np.load(P/'mapid.npy',mmap_mode='r');px=np.load(P/'pixel.npy',mmap_mode='r')
 base=np.load(P/'base.npy',mmap_mode='r');cached=np.load(P/'target.npy',mmap_mode='r')
 y=np.stack([p[mi,px//256,px%256] for p in ys],1).astype('float64')
 valid=np.stack([((p[mi,px//8]>>(7-px%8))&1).astype(bool) for p in ms],1)
 assert np.array_equal(valid,cached[:,2:]>0)
 assert np.max(np.abs(np.where(valid,y-cached[:,:2],0)))==0
 residual=y-base;tail=valid&(np.abs(residual)>=5)
 common=np.load(A/'nodes'/part/'rows.npy')
 assert np.array_equal(common,np.flatnonzero(tail.any(1)))
 territories,ti=np.unique(tm[mi],return_inverse=True);den=np.zeros((len(territories),2));non=np.zeros_like(den)
 np.add.at(den,ti,valid);np.add.at(non,ti,np.where(valid&~tail,residual**2,0))
 results[part]={}
 for mode in ['old','nodes','shuffled']:
  out=R/('runs/c130_nodes_'+mode);rows=np.load(out/(part+'_rows.npy'));assert np.array_equal(rows,common)
  prediction=np.load(out/(part+'_delta.npy')).astype('float64');prob=np.load(out/(part+'_prob.npy'))
  mask=tail[rows];r=residual[rows];err=np.where(mask,prediction-r,0)
  ss=(err**2).sum(0);n=mask.sum(0);field=non.copy();np.add.at(field,ti[rows],err**2)
  score=dict(tail_RMSE=np.sqrt(ss/n).tolist(),sign_accuracy=(((prediction*r>0)&mask).sum(0)/n).tolist(),
   classification_accuracy=((((prob>.5)==(r>0))&mask).sum(0)/n).tolist(),
   truth_gated_all_RMSE=np.sqrt(field.sum(0)/den.sum(0)).tolist(),tail_count=n.astype(int).tolist(),valid_count=den.sum(0).astype(int).tolist())
  original=json.loads((out/'result.json').read_text())['FIT' if part=='fit' else 'monitor']
  assert np.max(np.abs(np.array(score['tail_RMSE'])-original['tail_RMSE']))<2e-6
  assert np.max(np.abs(np.array(score['truth_gated_all_RMSE'])-original['truth_gated_all_RMSE']))<2e-6
  results[part][mode]=score
  if part=='monitor':monitor_stats[mode]=field
  print(part,mode,score['tail_RMSE'],score['truth_gated_all_RMSE'],flush=True)
 if part=='monitor':
  rng=np.random.default_rng(130501);bootstrap=rng.integers(len(territories),size=(2000,len(territories)));ci={}
  for other in ['old','shuffled']:
   gains=np.sqrt(monitor_stats[other][bootstrap].sum(1)/den[bootstrap].sum(1))-np.sqrt(monitor_stats['nodes'][bootstrap].sum(1)/den[bootstrap].sum(1))
   delta=np.sqrt(monitor_stats[other].sum(0)/den.sum(0))-np.sqrt(monitor_stats['nodes'].sum(0)/den.sum(0))
   ci[other]=dict(gain=delta.tolist(),paired_terrain_CI95=np.quantile(gains,[.025,.975],axis=0).T.tolist())
  np.savez(A/'probe_monitor_stats.npz',terrain_ids=territories,count=den,**monitor_stats)
out=dict(status='passed',source='originalTRAINlabels/packedmasks/NumPyFP64',results=results,comparison=ci,
 gate_passed=bool(np.all(np.array(ci['old']['gain'])>=.10) and np.all(np.array(ci['shuffled']['gain'])>0)),official_val_used=False,
 score_scope='oracle-known tail locations on sameC101fixed128terrainmonitor points;notdeployable')
(A/'probe_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
