"""Independent original-label scoring of completed capacity arms."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';A=R/'audit/c129_fitgap'
p=json.loads((A/'capacity_protocol.json').read_text())
ya=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ma=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
basefit=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
fitids=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());fitlookup={int(i):j for j,i in enumerate(fitids)}
basemon=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
monids=json.loads((R/'audit/c88_mesh/base_index.json').read_text())['ids'];monlookup={int(i):j for j,i in enumerate(monids)}
results={}
for tag in ['n16','n64','w64']:
 out=R/('runs/c129_capacity_'+tag)
 if not (out/'result.json').exists():continue
 original=json.loads((out/'result.json').read_text());results[tag]={}
 for label,filename,ids,ba,look in [('FIT','fit_prediction.npy',p['fit'+str(p['arms'][tag]['maps'])],basefit,fitlookup),('monitor','monitor_prediction.npy',p['monitor'],basemon,monlookup)]:
  pred=np.load(out/filename,mmap_mode='r');sums=np.zeros((6,2));rows=[]
  for start in range(0,len(ids),16):
   ix=ids[start:start+16];y=np.stack([a[ix] for a in ya],1).astype('float64')
   valid=np.stack([np.unpackbits(a[ix],axis=1).reshape(len(ix),256,256).astype(bool) for a in ma],1)
   base=np.asarray(ba[[look[int(i)] for i in ix]],dtype='float64')
   estimated=np.asarray(pred[start:start+len(ix)],dtype='float64')
   assert np.isfinite(estimated).all()
   error=np.where(valid,estimated-y,0);reference_error=np.where(valid,base-y,0)
   se=(error*error).sum((2,3));count=valid.sum((2,3))
   tail=valid&(np.abs(reference_error)>=5);tailerror=np.where(tail,error,0)
   direction=tail&((estimated-base)*(y-base)>0)
   sums+=np.array([se.sum(0),count.sum(0),(reference_error**2).sum((0,2,3)),(tailerror**2).sum((0,2,3)),tail.sum((0,2,3)),direction.sum((0,2,3))])
   rows.extend(np.concatenate([se,count],1).tolist())
  ss,nn,bs,ts,tn,correct=sums;rows=np.array(rows);permap=np.sqrt(rows[:,:2]/np.maximum(rows[:,2:],1))
  value=dict(RMSE=np.sqrt(ss/nn).tolist(),C82=np.sqrt(bs/nn).tolist(),valid=nn.astype(int).tolist(),
   tail_RMSE=np.sqrt(ts/tn).tolist(),tail_direction=(correct/tn).tolist(),worst_map_RMSE=permap.max(0).tolist(),
   maps_both_below1=int((permap.max(1)<1).sum()),maps=len(ids),exposures=original['exposures'],updates=original['updates'])
  assert np.max(np.abs(np.array(value['RMSE'])-original[label]['rmse']))<1e-5
  assert np.array_equal(value['valid'],original[label]['count'])
  results[tag][label]=value;np.save(out/(label.lower()+'_verified_by_map.npy'),rows)
  print(tag,label,value['RMSE'],flush=True)
out=dict(status='passed',source='originalTRAINlabels/packedmasks/NumPyfloat64',arms=results,official_val_used=False)
(A/'capacity_verified.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
