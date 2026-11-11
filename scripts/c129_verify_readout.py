from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data';A=R/'audit/c129_fitgap'
p=json.loads((R/'audit/c127_joint/protocol.json').read_text());out={}
ya=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ma=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
for name,filename,ids in [('parentFIT','fit_parent_prediction.npy',p['fit']),('closedFIT','fit_readout_prediction.npy',p['fit']),('closedMonitor','monitor_readout_prediction.npy',p['monitor'])]:
 pred=np.load(A/filename,mmap_mode='r');ss=np.zeros(2);nn=np.zeros(2)
 for start in range(0,len(ids),32):
  ix=ids[start:start+32];y=np.stack([a[ix] for a in ya],1).astype('float64')
  v=np.stack([np.unpackbits(a[ix],axis=1).reshape(len(ix),256,256).astype(bool) for a in ma],1)
  e=np.where(v,pred[start:start+len(ix)].astype('float64')-y,0);ss+=(e*e).sum((0,2,3));nn+=v.sum((0,2,3))
 out[name]=dict(RMSE=np.sqrt(ss/nn).tolist(),valid=nn.astype(int).tolist())
original=json.loads((A/'readout.json').read_text())
for a,b in [('parentFIT','parent_fullFIT'),('closedFIT','closed_fullFIT'),('closedMonitor','closed_monitor')]:
 assert np.max(np.abs(np.array(out[a]['RMSE'])-original[b]['RMSE']))<1e-5
out['status']='passed';out['source']='originalTRAINlabels/originalpackedmasks/NumPyfloat64';out['official_val_used']=False
(A/'readout_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
