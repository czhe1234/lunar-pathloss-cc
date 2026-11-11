"""CPU input-only C8 full-map export. No targets/masks are read.
Cached C3/C4 fields were computed only from heightmap and TX. Report cached
latency separately; this is not a claim that the end-to-end runtime limit is met.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from functools import lru_cache
import argparse,json,csv,time,sys,hashlib,shutil
import numpy as np,lightgbm as lgb
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c8_tx_context import terrain_fields,tx_features,NAMES
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['train','val'],default='train');p.add_argument('--part',choices=['dev','full'],default='dev');p.add_argument('--threads',type=int,default=16);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
run=R/'runs'/a.run;out=run/'predictions'/(f'{a.split}_{a.part}_cpu'+(f'_first{a.limit}' if a.limit else ''))
out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
for src in [Path(__file__),R/'src/c8_tx_context.py']:shutil.copy2(src,out/'source'/src.name)
models=[lgb.Booster(model_file=str(run/f'band{b}.txt')) for b in ['415','58']]
assert all(m.num_feature()==104 for m in models)
rows=list(csv.DictReader((R.parent/'data'/f'{a.split}_index.csv').open()));H=np.load(R.parent/'data'/f'{a.split}_hm.npy',mmap_mode='r')
G=[np.load(R/f'features/{name}/{a.split}.npy',mmap_mode='r') for name in ['c3','c4_extra']]
ids=np.arange(len(rows))
if a.part=='dev':
 assert a.split=='train';dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids']);ids=np.array([i for i,r in enumerate(rows) if r['terrain_id'] in dev])
if a.limit:ids=ids[:a.limit]
mapping={'source_split':a.split,'index_rows':ids.tolist(),'sample_ids':[rows[i]['sample_id'] for i in ids]};(out/'indices.json').write_text(json.dumps(mapping,indent=2)+'\n')
yy,xx=np.mgrid[:256,:256];yy=yy.astype(np.float32);xx=xx.astype(np.float32)
@lru_cache(maxsize=32)
def fields(i):
 h=np.array(H[i],np.float32);tf=terrain_fields(h);hp=h-uniform_filter(h,size=5,mode='nearest')
 return h,tf,hp
def inputs(i):
 row=rows[i];tr=int(row['tx_row']);tc=int(row['tx_col']);h,tf,hp=fields(int(row['hm_row']));sx=tf[1];sy=tf[2];ht=float(h[tr,tc]);dx=xx-tc;dy=yy-tr;dz=h+1-ht-3;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
 basic=np.stack([h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((yy==tr)&(xx==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)],0).reshape(11,-1).T
 gg=np.concatenate([np.asarray(g[i],np.float32) for g in G],0).reshape(20,-1).T
 ctx=tx_features(tf,tr,tc);x=np.concatenate((basic,gg,np.broadcast_to(ctx,(65536,len(ctx)))),1)
 fs=(20*np.log10(4*np.pi*dist.reshape(-1,1)*np.array([415e6,5.775e9])[None]/299792458.)).astype(np.float32)
 return x,fs
arr=[np.lib.format.open_memmap(out/f'pl{b}.npy',mode='w+',dtype=np.float32,shape=(len(ids),256,256)) for b in ['415','58']]
start=time.time();times=[]
for j,i in enumerate(ids):
 t=time.perf_counter();x,fs=inputs(int(i));pr=np.stack([m.predict(x,num_threads=a.threads).astype(np.float32)+fs[:,k] for k,m in enumerate(models)],1);assert np.isfinite(pr).all()
 for k in [0,1]:arr[k][j]=pr[:,k].reshape(256,256)
 times.append(time.perf_counter()-t)
 if j%64==0:print(json.dumps({'done':j+1,'total':len(ids),'seconds':time.time()-start,'last_map_seconds':times[-1]}),flush=True)
for v in arr:v.flush()
meta={'run':a.run,'split':a.split,'part':a.part,'maps_per_band':len(ids),'precision':'CPU LightGBM FP64 tree accumulation; FP32 export and free-space reference','no_target_or_validity_mask_read':True,'input_fields':'Heightmap, TX, previously verified input-only C3/C4 caches, TX context','GPU_used':False,'threads':a.threads,'seconds':time.time()-start,'cached_feature_map_seconds_mean':float(np.mean(times)),'cached_feature_map_seconds_p95':float(np.percentile(times,95)),'latency_warning':'Excludes constructing C3/C4 geometry caches. Not end-to-end official-runtime compliance.','model_sha256':{b:hashlib.sha256((run/f'band{b}.txt').read_bytes()).hexdigest() for b in ['415','58']}}
(out/'inference.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

