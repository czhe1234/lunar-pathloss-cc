"""TRAIN-side label sensitivity to 0,1,and up to2m TX movement; no fitting."""
import csv,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path.cwd();D=R.parent/'data';A=R/'audit/c111_TXcoverage';t=time.monotonic()
rows=list(csv.DictReader(open(D/'train_index.csv')));fit=set(np.load(R/'features/c38_dense/fit_map_indices.npy').tolist());by=defaultdict(list)
for i in fit:by[rows[i]['terrain_id']].append(i)
yy=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
ww=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
gx,gy=np.meshgrid(np.arange(256),np.arange(256));record=[]
for terrain,ids in sorted(by.items()):
 if len(ids)<2:continue
 ids=sorted(ids);pos=np.array([[int(rows[i]['tx_row']),int(rows[i]['tx_col'])] for i in ids]);ds=((pos[:,None]-pos[None])**2).sum(-1)
 for a,b in zip(*np.where(np.triu(ds<=4,1))):
  i,j=ids[a],ids[b];d2=int(ds[a,b]);cat='exact0' if d2==0 else 'unit1m' if d2==1 else 'near_sqrt2to2m'
  remote=((gy-pos[a,0])**2+(gx-pos[a,1])**2>=100)&((gy-pos[b,0])**2+(gx-pos[b,1])**2>=100)
  bands=[]
  for k in range(2):
   w=np.unpackbits(ww[k][i]).reshape(256,256).astype(bool)&np.unpackbits(ww[k][j]).reshape(256,256).astype(bool)&remote
   delta=(yy[k][i].astype(np.float64)-yy[k][j].astype(np.float64))[w]
   bands.append(dict(count=len(delta),sse=float((delta**2).sum()),change_over1=int((abs(delta)>1).sum()),change_over5=int((abs(delta)>5).sum()),
    abs_quantiles=np.quantile(abs(delta),[.5,.9,.99,1]).tolist() if len(delta) else []))
  record.append(dict(terrain=terrain,ids=[i,j],distance_m=float(np.sqrt(d2)),category=cat,bands=bands))
summary={}
for cat in ['exact0','unit1m','near_sqrt2to2m']:
 sub=[z for z in record if z['category']==cat];out=[]
 for b in range(2):
  count=sum(z['bands'][b]['count'] for z in sub);sse=sum(z['bands'][b]['sse'] for z in sub)
  out.append(dict(count=count,RMSE_label_difference=float(np.sqrt(sse/max(count,1))),fraction_abschange_gt1=sum(z['bands'][b]['change_over1'] for z in sub)/max(count,1),
   fraction_abschange_gt5=sum(z['bands'][b]['change_over5'] for z in sub)/max(count,1)))
 summary[cat]=dict(pairs=len(sub),terrains=len(set(z['terrain'] for z in sub)),bands=out)
result=dict(summary=summary,perpair=record,scope='FITlabels only;receiver>=10mfrombothTX;pairedcommonvalidpixels;notpredictionorinformationfloor',
 interpretation='RawfieldTXsensitivityonly;doesnotbyitselfshowC82residualderivativesarewrong',seconds=time.monotonic()-t,official_VAL_used=False)
(A/'nearTX_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
