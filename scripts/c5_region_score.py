"""Independent region-wise score on existing development predictions (CPU)."""
from pathlib import Path
import argparse,json
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);args=p.parse_args();P=R/args.predictions
mapping=json.loads((P/'indices.json').read_text());assert mapping['source_split']=='train'
ids=np.array(mapping['index_rows']);g=np.load(R/'features/c3/train.npy',mmap_mode='r')
regions=['both_clear_geometry','height_sensitive_geometry','blocked_at_3m_geometry'];out={}
for band in ['415','58']:
 pred=np.load(P/f'pl{band}.npy',mmap_mode='r');y=np.load(D/f'train_pl{band}.npy',mmap_mode='r');m=np.load(D/f'train_mask{band}.npy',mmap_mode='r');ss=np.zeros(3);nn=np.zeros(3,dtype=np.int64)
 for lo in range(0,len(ids),8):
  ii=ids[lo:lo+8];e3=(2*np.sinh(3*g[ii,:8].astype(np.float32))).max(1);e2=2*np.sinh(3*g[ii,8].astype(np.float32));v=np.unpackbits(m[ii],axis=1,bitorder='big').reshape(len(ii),256,256).astype(bool)
  e=(pred[lo:lo+len(ii)].astype(np.float64)-y[ii].astype(np.float64))**2
  group=[(e3<0)&(e2<0),(e3<0)&(e2>=0),e3>=0]
  assert np.all(np.stack(group).sum(0)==1)
  for j,mask in enumerate(group):
   select=v&mask;ss[j]+=e[select].sum();nn[j]+=select.sum()
 out[band]={name:{'sse':float(ss[j]),'count':int(nn[j]),'rmse':float(np.sqrt(ss[j]/nn[j])),'fraction_of_all_sse':float(ss[j]/ss.sum())} for j,name in enumerate(regions)}
result={'predictions':args.predictions,'split':'development','maps_per_band':len(ids),'regions_are_input_geometry_hypotheses_not_true_path_mechanism_labels':True,'bands':out}
(P/'region_score.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
