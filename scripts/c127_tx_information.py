"""TRAIN-only diagnostic: do other observed TX residuals predict a held-out TX residual?"""
import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';A=R/'audit/c127_joint'
p=json.loads((A/'protocol.json').read_text());rows=list(csv.DictReader(open(D/'train_index.csv')))
ya=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ma=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
ix=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());lookup={int(i):j for j,i in enumerate(ix)}
base=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
tot=np.zeros((2,8),np.float64);dist=[]
for terrain,positions in p['groups'].items():
 if len(positions)!=4:continue
 ids=[p['fit'][j] for j in positions];b=np.asarray(base[[lookup[i] for i in ids]],np.float64)
 y=np.stack([v[ids] for v in ya],1).astype(np.float64);valid=np.stack([np.unpackbits(v[ids],axis=1).reshape(4,256,256).astype(bool) for v in ma],1)
 e=np.where(valid,y-b,0);num=valid.sum(0);total=e.sum(0)
 for j in range(4):
  have=(num-valid[j])>=1;v=valid[j]&have;other=(total-e[j])/np.maximum(num-valid[j],1)
  corrected=np.where(v,e[j]-other,0);true_group=(total/np.maximum(num,1))
  oracle=np.where(v,e[j]-true_group,0);r=np.where(v,e[j],0)
  tail=v&(np.abs(e[j])>=5)
  tot[:,0]+=(r*r).sum((1,2));tot[:,1]+=(corrected*corrected).sum((1,2));tot[:,2]+=(oracle*oracle).sum((1,2));tot[:,3]+=v.sum((1,2))
  tot[:,4]+=(tail&(other*e[j]>0)).sum((1,2));tot[:,5]+=tail.sum((1,2))
  tot[:,6]+=np.where(v,other*e[j],0).sum((1,2));tot[:,7]+=np.where(v,other*other,0).sum((1,2))
 for i in range(4):
  for j in range(i):
   a=rows[ids[i]];brow=rows[ids[j]]
   dist.append(np.hypot(int(a['tx_row'])-int(brow['tx_row']),int(a['tx_col'])-int(brow['tx_col'])))
out=dict(scope='277 FIT terrains,4TX each; true other-TX labels used only for diagnosis,not deployment or training change',
 reference_RMSE=np.sqrt(tot[:,0]/tot[:,3]).tolist(),subtract_other3_mean_RMSE=np.sqrt(tot[:,1]/tot[:,3]).tolist(),
 subtract_own_inclusive4_mean_RMSE=np.sqrt(tot[:,2]/tot[:,3]).tolist(),tail_otherTX_direction_accuracy=(tot[:,4]/tot[:,5]).tolist(),
 cross_moment=(tot[:,6]/tot[:,3]).tolist(),other_mean_second_moment=(tot[:,7]/tot[:,3]).tolist(),
 TX_distance_quantiles=np.quantile(dist,[.1,.5,.9]).tolist(),valid_count=tot[:,3].astype(int).tolist(),official_val_used=False)
(A/'tx_information.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
