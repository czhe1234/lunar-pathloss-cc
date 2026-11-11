"""Independent full-map CPU scoring, two implementations, no GPU context."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,json,csv,time
import numpy as np
import torch
torch.set_num_threads(8)
R=Path(__file__).resolve().parents[1];D=R.parent/'data';p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);a=p.parse_args();out=R/a.predictions
mapping=json.loads((out/'indices.json').read_text());meta=json.loads((out/'inference.json').read_text());split=mapping['source_split'];ids=np.array(mapping['index_rows']);rows=list(csv.DictReader((D/f'{split}_index.csv').open()))
assert len(ids)==meta['maps_per_band'] and len(set(ids.tolist()))==len(ids)
assert [rows[i]['sample_id'] for i in ids]==mapping['sample_ids']
results={};permap={};start=time.time()
for b in ['415','58']:
 pred=np.load(out/f'pl{b}.npy',mmap_mode='r');truth=np.load(D/f'{split}_pl{b}.npy',mmap_mode='r');mask=np.load(D/f'{split}_mask{b}.npy',mmap_mode='r')
 assert pred.dtype==np.float32 and pred.shape==(len(ids),256,256)
 total=0.;count=0;ts=0.;tc=0;alltotal=0.;groups={g:[0.,0,0] for g in ['single','multi']}
 for lo in range(0,len(ids),8):
  ii=ids[lo:lo+8];pr=pred[lo:lo+len(ii)].astype(np.float64);y=truth[ii].astype(np.float64);m=np.unpackbits(mask[ii],axis=1,bitorder='big').reshape(len(ii),256,256).astype(bool)
  assert np.isfinite(pr).all()
  e=(pr-y)**2;ss=np.where(m,e,0).sum((1,2));nn=m.sum((1,2));total+=float(ss.sum());count+=int(nn.sum());alltotal+=float(e.sum())
  # Recompute the subtraction and squaring in Torch CPU, not from NumPy errors.
  te=(torch.from_numpy(pr)-torch.from_numpy(y)).square();tm=torch.from_numpy(m)
  ts+=float(te.masked_select(tm).sum());tc+=int(tm.sum())
  for j,i in enumerate(ii):
   r=rows[i];g=groups[r['group']];g[0]+=float(ss[j]);g[1]+=int(nn[j]);g[2]+=1
   rec=permap.setdefault(int(i),{'index_row':int(i),'sample_id':r['sample_id'],'terrain_id':r['terrain_id'],'group':r['group']});rec[f'sse{b}']=float(ss[j]);rec[f'count{b}']=int(nn[j])
 rm=(total/count)**.5;trm=(ts/tc)**.5;assert count==tc and abs(rm-trm)<1e-8
 results[b]={'rmse_db':rm,'sse':total,'valid_pixels':count,'torch_CPU_rmse_db':trm,'same_prediction_scorer_difference_db':abs(rm-trm),'all_pixel_rmse_diagnostic':(alltotal/(len(ids)*65536))**.5,'groups':{g:{'rmse_db':(v[0]/v[1])**.5,'valid_pixels':v[1],'maps':v[2]} for g,v in groups.items() if v[1]}}
full=len(ids)==len(rows) if split=='val' else meta['part']=='dev' and len(ids)==2562
result={'evaluation_split':'official_validation' if split=='val' else meta['part'],'source_split':split,'maps_per_band':len(ids),'complete_requested_split':full,'bands':results,'goal_met_on_official_validation':split=='val' and full and all(v['rmse_db']<=1 for v in results.values()),'calculation':'Each band: pooled FP64 squared error over official valid masks, divided by pooled count, square root. Independent NumPy and Torch CPU.','seconds':time.time()-start,'scoring_GPU_used':False}
ref=R/'runs'/('c4_full_refit_s20261031/predictions/val_full_bf16' if split=='val' else 'c4_critical_length_fast_s20261031/predictions/train_dev_bf16')
if full and (ref/'independent_score.json').exists():
 old=json.loads((ref/'independent_score.json').read_text());assert all(old['bands'][b]['valid_pixels']==results[b]['valid_pixels'] for b in results)
 result['C4_reference']={b:{'rmse_db':old['bands'][b]['rmse_db'],'relative_RMSE_reduction_percent':100*(1-results[b]['rmse_db']/old['bands'][b]['rmse_db'])} for b in results}
(out/'independent_score.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'error_by_map.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(next(iter(permap.values()))));w.writeheader();w.writerows(permap.values())
print(json.dumps(result),flush=True)

