"""Label-informed geometric hypothesis scan. No learned predictor or weight updates."""
from pathlib import Path
import numpy as np,csv,json,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/c4_remaining_error_20261031';DATA=ROOT.parent/'data';rows=list(csv.DictReader((DATA/'train_index.csv').open()));dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids']);selection=json.loads((OUT/'selection.json').read_text());fitids=set(selection['train_indices']);geo=np.load(ROOT/'features/c4_extra/train.npy',mmap_mode='r');hm=np.load(DATA/'train_hm.npy',mmap_mode='r');ys=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ms=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
rxr,rxc=np.meshgrid(np.arange(4,256,8),np.arange(4,256,8),indexing='ij');rr=rxr.ravel();cc=rxc.ravel();heights=np.arange(.25,4.001,.025);records=[];curves=[];t0=time.time()
ids=[i for i,r in enumerate(rows) if r['group']=='multi' and (r['terrain_id'] in dev or i in fitids)]
for i in ids:
 row=rows[i];h=hm[int(row['hm_row'])];tr=int(row['tx_row']);tc=int(row['tx_col']);ht=float(h[tr,tc]);need=2*np.sinh(geo[i,0,rr,cc].astype(float)*3)+2.5;distance=np.sqrt((rr-tr)**2+(cc-tc)**2+(h[rr,cc]+1-ht-3)**2);curve=[];bands=[]
 for b,f in enumerate([415e6,5.775e9]):
  fs=20*np.log10(4*np.pi*distance*f/299792458.);y=ys[b][i,rr,cc].astype(float)-fs;v=np.unpackbits(ms[b][i],bitorder='big').reshape(256,256)[rr,cc].astype(bool);cl=v&(y<3);bl=v&(y>10)
  if min(cl.sum(),bl.sum())<8:curve.append(np.full(len(heights),np.nan));bands.append(None);continue
  cs=np.sort(need[cl]);bs=np.sort(need[bl]);ba=.5*(np.searchsorted(cs,heights,side='left')/len(cs)+1-np.searchsorted(bs,heights,side='left')/len(bs));curve.append(ba);best=np.argmax(ba);bands.append({'clear_n':int(cl.sum()),'blocked_n':int(bl.sum()),'aopt':float(heights[best]),'max_BA':float(ba[best]),'near_optimal_width':float(np.ptp(heights[ba>=ba[best]-.005]))})
 records.append({'index_row':i,'terrain_id':row['terrain_id'],'cohort':'dev_discovery' if row['terrain_id'] in dev else 'train_confirmation','tx_height_relative':ht,'tx_row':tr,'tx_col':tc,'bands':bands});curves.append(curve)
C=np.array(curves);H=np.array([a['tx_height_relative'] for a in records]);np.savez_compressed(OUT/'train_height_curves.npz',curves=C,heights=heights,tx_height=H,index_rows=np.array(ids));(OUT/'train_height_records.json').write_text(json.dumps(records,indent=2)+'\n')
# Phase inferred from every second TX; evaluate other TX in that SAME terrain.
results=[]
for tid in sorted({a['terrain_id'] for a in records}):
 js=np.array([j for j,a in enumerate(records) if a['terrain_id']==tid]);cal=js[::2];test=js[1::2];rec={'terrain_id':tid,'cohort':records[js[0]]['cohort'],'models':{}}
 def evaluate_pred(jj,pred):
  a=np.asarray(pred);a=np.broadcast_to(a,(len(jj),));q=np.clip(np.rint((a-heights[0])/.025).astype(int),0,len(heights)-1);return C[jj,:,q]
 baseline=evaluate_pred(test,3.);rec['models']['height3']={'BA':np.nanmean(baseline,axis=0).tolist()};rec['models']['height2']={'BA':np.nanmean(evaluate_pred(test,2.),axis=0).tolist()}
 mean=np.nanmean(C[cal],axis=(0,1));a=heights[np.nanargmax(mean)];rec['models']['terrain_constant']={'a':float(a),'BA':np.nanmean(evaluate_pred(test,a),axis=0).tolist()}
 for unit in [.25,.5,1.,2.]:
  phases=np.arange(100)/100*unit;vals=[]
  for phase in phases:
   pred=3.-np.mod(H[cal]+phase,unit);vals.append(np.nanmean(evaluate_pred(cal,pred)))
  phi=float(phases[np.nanargmax(vals)]);pred=3.-np.mod(H[test]+phi,unit);rec['models'][f'floor_q{unit}']={'phase':phi,'BA':np.nanmean(evaluate_pred(test,pred),axis=0).tolist()}
 rec['label_height_by_band']=[{'std':float(np.nanstd([records[j]['bands'][b]['aopt'] if records[j]['bands'][b] else np.nan for j in js]))} for b in range(2)];results.append(rec)
summary={}
for cohort in ['dev_discovery','train_confirmation']:
 group=[a for a in results if a['cohort']==cohort];summary[cohort]={name:np.nanmean([a['models'][name]['BA'] for a in group],axis=0).tolist() for name in group[0]['models']}
(OUT/'phase_hypothesis_train.json').write_text(json.dumps({'summary':summary,'per_terrain':results,'seconds':time.time()-t0,'scope':'Geometric hypothesis calibration with known labels for every second TX; held other TX within each terrain. Not a deployable predictor.'},indent=2)+'\n');print(json.dumps(summary),flush=True)
