import json,csv,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'data';rows=list(csv.DictReader((DATA/'train_index.csv').open()));dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids']);rng=np.random.default_rng(20261031)
hm=np.load(DATA/'train_hm.npy',mmap_mode='r');ys=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ms=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
rr,cc=np.meshgrid(np.arange(4,256,6),np.arange(4,256,6),indexing='ij');r=rr.ravel();c=cc.ravel();fold=(rr//6+cc//6).ravel()%2;t=(np.arange(256,dtype=np.float64)+.5)/256;heights=np.arange(.25,4.001,.025);results=[];t0=time.time()
for cohort in ['fit','dev']:
 ids=rng.choice([i for i,a in enumerate(rows) if (a['terrain_id'] in dev)==(cohort=='dev')],128,replace=False)
 for i in ids:
  row=rows[i];h=hm[int(row['hm_row'])].astype(float);tr=int(row['tx_row']);tc=int(row['tx_col']);px=tc+(c[None]-tc)*t[:,None];py=tr+(r[None]-tr)*t[:,None];r0=np.floor(py).astype(int);c0=np.floor(px).astype(int);r1=np.minimum(255,r0+1);c1=np.minimum(255,c0+1);a=py-r0;b=px-c0
  hh=h[r0,c0]*(1-a)*(1-b)+h[r1,c0]*a*(1-b)+h[r0,c1]*(1-a)*b+h[r1,c1]*a*b
  need=((hh-h[tr,tc]*(1-t[:,None])-(h[r,c]+1)*t[:,None])/(1-t[:,None])).max(0)
  distance=np.sqrt((r-tr)**2+(c-tc)**2+(h[r,c]+1-h[tr,tc]-3)**2)
  bands=[]
  for k,freq in enumerate([415e6,5.775e9]):
   fs=20*np.log10(4*np.pi*distance*freq/299792458.);v=np.unpackbits(ms[k][i],bitorder='big').reshape(256,256)[r,c].astype(bool);ex=ys[k][i,r,c].astype(float)-fs;clear=v&(ex<3);blocked=v&(ex>10)
   if min((clear&(fold==0)).sum(),(blocked&(fold==0)).sum(),(clear&(fold==1)).sum(),(blocked&(fold==1)).sum())<8:bands.append(None);continue
   pred=need[None]<heights[:,None];f=fold==0;ba=.5*((pred&(clear&f)).sum(1)/(clear&f).sum()+((~pred)&(blocked&f)).sum(1)/(blocked&f).sum());aopt=float(heights[np.argmax(ba)])
   def score(a,part):
    p=need<a;cl=clear&(fold==part);bl=blocked&(fold==part)
    return {'n':int((cl|bl).sum()),'errors':int(((~p)&cl).sum()+(p&bl).sum()),'balanced_accuracy':float(.5*((p&cl).sum()/cl.sum()+((~p)&bl).sum()/bl.sum()))}
   bands.append({'aopt_from_half_receivers':aopt,'check_half_fixed2':score(2.,1),'check_half_oracle':score(aopt,1)})
  sy,sx=np.gradient(h);results.append({'cohort':cohort,'index_row':int(i),'terrain_id':row['terrain_id'],'tx_row':tr,'tx_col':tc,'tx_slope_row':float(sy[tr,tc]),'tx_slope_col':float(sx[tr,tc]),'tx_height':float(h[tr,tc]),'bands':bands})
summary={}
for cohort in ['fit','dev']:
 summary[cohort]=[]
 for k in range(2):
  b=[r['bands'][k] for r in results if r['cohort']==cohort and r['bands'][k] is not None];aa=[v['aopt_from_half_receivers'] for v in b]
  summary[cohort].append({'maps':len(b),'aopt_quantiles':np.quantile(aa,[0,.1,.5,.9,1]).tolist(),'fixed2_errors':sum(v['check_half_fixed2']['errors'] for v in b),'oracle_errors':sum(v['check_half_oracle']['errors'] for v in b),'pixels':sum(v['check_half_oracle']['n'] for v in b),'fixed2_mean_BA':float(np.mean([v['check_half_fixed2']['balanced_accuracy'] for v in b])),'oracle_mean_BA':float(np.mean([v['check_half_oracle']['balanced_accuracy'] for v in b]))})
out=ROOT/'audit/c3_height_calibration_audit.json';out.write_text(json.dumps({'summary':summary,'records':results,'seconds':time.time()-t0,'scope':'Label-informed per-map calibration diagnostic. Threshold fit and check on alternating RX subsets. Not deployable or official validation.'},indent=2)+'\n');print(json.dumps(summary),flush=True)
