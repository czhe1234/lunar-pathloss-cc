"""Freeze input-only selection and compare analytic resampling/smoothing controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import convolve,median_filter
R=Path(__file__).resolve().parents[1];O=R/'audit/c19_fine_physics_20261101';D=R.parent/'data'
rows=list(csv.DictReader((D/'train_index.csv').open()));H=np.load(D/'train_hm.npy',mmap_mode='r')
c18=json.loads((R/'audit/c18_identifiability_20261101/residual_probe/manifest.json').read_text());cal=set(c18['calibration_terrains']);confirm=set(c18['confirmation_terrains']);terr=sorted(cal|confirm)
by={}
for i,r in enumerate(rows):
 if r['terrain_id'] in cal|confirm:by.setdefault(r['terrain_id'],[]).append(i)
rng=np.random.default_rng(20261108);ids=[int(rng.choice(by[t])) for t in terr]
Pdir=R/'runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh'
pm=json.loads((Pdir/'indices.json').read_text());pos={i:j for j,i in enumerate(pm['index_rows'])}
P=[np.load(Pdir/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
G=np.load(R/'features/c3/train.npy',mmap_mode='r')
pixels=[]
allowed=(np.arange(2,254)[:,None]*256+np.arange(2,254)[None]).ravel()
for i in ids:pixels.append(np.random.default_rng(20261109+104729*i).choice(allowed,256,replace=False))
pixels=np.array(pixels)
selection={'index_rows':ids,'terrain_ids':terr,'parts':['calibration' if t in cal else 'confirmation' for t in terr],'pixels_per_map':256,'input_only_selection':True,'C10_prediction_directory':str(Pdir)}
(O/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');np.save(O/'pixels.npy',pixels)
K=np.outer([1,2,1],[1,2,1])/16;yy,xx=np.mgrid[:256,:256]
names=['C10','db_binomial3','relative_power_binomial3','excess_median3','geometry_matched_box3']
start=time.time()
def one(pair):
 j,i=pair;row=rows[i];h=np.asarray(H[int(row['hm_row'])],float);tr=int(row['tx_row']);tc=int(row['tx_col'])
 d=np.sqrt((xx-tc)**2+(yy-tr)**2+(h+1-h[tr,tc]-3)**2).clip(.25);fs=20*np.log10(4*np.pi*d[None]*np.array([415e6,5.775e9])[:,None,None]/299792458.)
 g=np.asarray(G[i],np.float32);e3=(2*np.sinh(3*g[:8])).max(0);e2=2*np.sinh(3*g[8]);zones={'all':np.ones((256,256),bool),'occluded_proxy':e2>=0,'height_sensitive':(e3<0)&(e2>=0),'nominal_blocked':e3>=0}
 ps=np.stack([p[pos[i]] for p in P]).astype(float);ys=np.stack([y[i] for y in Y]).astype(float);ms=np.stack([np.unpackbits(m[i],bitorder='big').reshape(256,256) for m in M]).astype(bool);ex=ps-fs
 result={'index_row':i,'part':selection['parts'][j],'terrain_id':terr[j],'metrics':{}}
 pred={'C10':ps,'db_binomial3':np.stack([convolve(v,K,mode='nearest') for v in ex])+fs,
 'relative_power_binomial3':np.stack([-10*np.log10(convolve(10**(-np.clip(v,-30,200)/10),K,mode='nearest').clip(1e-30)) for v in ex])+fs,
 'excess_median3':np.stack([median_filter(v,size=3,mode='nearest') for v in ex])+fs}
 # Preserve geometry state and use fixed0.1m critical-clearance matching.
 num=ex.copy();den=np.ones((256,256));state=e2>=0
 for dr in [-1,0,1]:
  for dc in [-1,0,1]:
   if dr==0 and dc==0:continue
   ri=np.clip(np.arange(256)+dr,0,255);ci=np.clip(np.arange(256)+dc,0,255)
   en=e2[ri[:,None],ci[None]];w=np.exp(-np.minimum(((en-e2)/.1)**2,50)/2)*(state==(en>=0))
   num+=ex[:,ri[:,None],ci[None]]*w;den+=w
 pred['geometry_matched_box3']=num/den+fs
 for name,pr in pred.items():
  result['metrics'][name]={}
  for zn,z in zones.items():
   result['metrics'][name][zn]=[{'sse':float(((pr[k]-ys[k])**2)[ms[k]&z].sum()),'n':int((ms[k]&z).sum())} for k in [0,1]]
 # Local curvature fields: query minus mean four diagonal neighbours.
 rr,cc=np.divmod(pixels[j],256);r=np.stack((rr-1,rr-1,rr+1,rr+1));c=np.stack((cc-1,cc+1,cc-1,cc+1))
 sy=ys[:,r,c].mean(1).T;sp=ps[:,r,c].mean(1).T;qy=ys[:,rr,cc].T;qp=ps[:,rr,cc].T
 qmask=(ms[:,rr,cc]&ms[:,r,c].all(1)).T
 sample={'Y':qy,'P':qp,'QY':qy-sy,'QP':qp-sp,'mask':qmask,'zone':np.where(e3[rr,cc]>=0,2,np.where(e2[rr,cc]>=0,1,0)),
 'context':np.stack((rr,cc,rr-tr,cc-tc,d[rr,cc],e3[rr,cc],e2[rr,cc]),1)}
 return result,sample
with ThreadPoolExecutor(max_workers=8) as pool:done=list(pool.map(one,enumerate(ids)))
records=[v[0] for v in done];sample={k:np.stack([v[1][k] for v in done]) for k in done[0][1]}
np.savez(O/'diagnostic_samples.npz',**sample)
stats={};calmask=np.array(selection['parts'])=='calibration';confmask=~calmask;rng=np.random.default_rng(20261110);w=rng.multinomial(306,np.full(306,1/306),size=1000)
for name in names:
 stats[name]={}
 for zn in records[0]['metrics'][name]:
  stats[name][zn]={}
  for k,b in enumerate(['415','58']):
   ss=np.array([r['metrics'][name][zn][k]['sse'] for r in records]);nn=np.array([r['metrics'][name][zn][k]['n'] for r in records]);base=np.array([r['metrics']['C10'][zn][k]['sse'] for r in records])
   stats[name][zn][b]={'calibration_RMSE':float(np.sqrt(ss[calmask].sum()/nn[calmask].sum())),'confirmation_RMSE':float(np.sqrt(ss[confmask].sum()/nn[confmask].sum())),
    'confirmation_valid_pixels':int(nn[confmask].sum()),'confirmation_SSE_reduction_percent':float(100*(1-ss[confmask].sum()/base[confmask].sum())),
    'terrain_bootstrap95':np.percentile(100*(1-(w@ss[confmask])/(w@base[confmask])),[2.5,97.5]).tolist()}
out={'state':'completed','scope':'612 internalDEV terrains, one input-selectedTX each,306/306 calibration/confirmation; full-map masked scores for fixed filters. No fit/tuning.',
 'statistics':stats,'seconds':time.time()-start,'selection':'selection.json','raw_statistics':records}
(O/'filter_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'state':'completed','seconds':out['seconds'],'occluded':{n:v['occluded_proxy'] for n,v in stats.items()}}),flush=True)
