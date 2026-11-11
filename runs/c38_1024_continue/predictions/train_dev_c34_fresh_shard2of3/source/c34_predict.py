"""C34 input-only full-map export. No targets/masks are read.
Cached C3/C4 fields were computed only from heightmap and TX. Report cached
latency separately; this is not a claim that the end-to-end runtime limit is met.
"""
import os
from pathlib import Path
from functools import lru_cache
import argparse,json,csv,time,sys,hashlib,shutil
import numpy as np,lightgbm as lgb
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c8_tx_context import terrain_fields,tx_features,NAMES
from src.cc_core import gpu_guard
import torch
from src.c34_model import Model
from src.c10_phase_geometry import phase_features,phase_features_gpu
from src.c9_extra_gpu import profile_gpu,surface_gpu
from src.c7_surface_features import tx_visibility,terrain_fields as surface_fields
info=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8)
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['train','val'],default='train');p.add_argument('--part',choices=['dev','full'],default='dev');p.add_argument('--threads',type=int,default=16);p.add_argument('--limit',type=int,default=0);p.add_argument('--fresh',action='store_true');p.add_argument('--save-zones',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
run=R/'runs'/a.run;out=run/'predictions'/(f'{a.split}_{a.part}_c34'+('_fresh' if a.fresh else '')+(f'_first{a.limit}' if a.limit else '')+f'_shard{a.shard}of{a.shards}')
out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
for src in [Path(__file__),R/'src/c8_tx_context.py',R/'src/c34_model.py',R/'src/c3_geometry.py',R/'src/c4_geometry.py',R/'src/c9_extra_gpu.py',R/'src/c7_surface_features.py',R/'src/c10_phase_geometry.py']:shutil.copy2(src,out/'source'/src.name)
ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False);cfg=ck['config'];st=ck['model']
if cfg.get('expanded',False):
 from src.c36_model import Model
net=Model(cfg['mode'],cfg['members'],cfg['width'],st['embedding.mean'],st['embedding.std'],st['embedding.edges'],st['bias'][0]*20).cuda()
net.load_state_dict(st);net.eval();nf=294
torch.set_float32_matmul_precision('high')
@torch.inference_mode()
def predict(xx,fs):
 out=[]
 for lo in range(0,len(xx),4096):
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(xx[lo:lo+4096]).mean(1)
  out.append(q.float().cpu().numpy()*20)
 return np.concatenate(out)+fs
if a.fresh:
 sys.path.insert(0,str(R/'src'));from c4_geometry import geometry,extra_geometry
rows=list(csv.DictReader((R.parent/'data'/f'{a.split}_index.csv').open()));H=np.load(R.parent/'data'/f'{a.split}_hm.npy',mmap_mode='r')
G=[] if a.fresh else [np.load(R/f'features/{name}/{a.split}.npy',mmap_mode='r') for name in ['c3','c4_extra']]
ids=np.arange(len(rows))
if a.part=='dev':
 assert a.split=='train';dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids']);ids=np.array([i for i,r in enumerate(rows) if r['terrain_id'] in dev])
if a.limit:ids=ids[:a.limit]
ids=ids[a.shard::a.shards]
mapping={'source_split':a.split,'index_rows':ids.tolist(),'sample_ids':[rows[i]['sample_id'] for i in ids]};(out/'indices.json').write_text(json.dumps(mapping,indent=2)+'\n')
yy,xx=np.mgrid[:256,:256];yy=yy.astype(np.float32);xx=xx.astype(np.float32)
@lru_cache(maxsize=32)
def fields(i):
 h=np.array(H[i],np.float32);tf=terrain_fields(h);hp=h-uniform_filter(h,size=5,mode='nearest')
 return h,tf,hp
def inputs(i):
 row=rows[i];tr=int(row['tx_row']);tc=int(row['tx_col']);h,tf,hp=fields(int(row['hm_row']));sx=tf[1];sy=tf[2];ht=float(h[tr,tc]);dx=xx-tc;dy=yy-tr;dz=h+1-ht-3;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
 basic=np.stack([h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((yy==tr)&(xx==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)],0).reshape(11,-1).T
 if a.fresh:
  with torch.inference_mode():
   hh=torch.from_numpy(h[None]).cuda();rr=torch.tensor([tr],device='cuda');cc=torch.tensor([tc],device='cuda');fg=torch.cat((geometry(hh,rr,cc),extra_geometry(hh,rr,cc)),1)[0].float().cpu().numpy()
  gg=fg.reshape(20,-1).T
 else:gg=np.concatenate([np.asarray(g[i],np.float32) for g in G],0).reshape(20,-1).T
 ctx=tx_features(tf,tr,tc);x=np.concatenate((basic,gg,np.broadcast_to(ctx,(65536,len(ctx)))),1)
 fs=(20*np.log10(4*np.pi*dist.reshape(-1,1)*np.array([415e6,5.775e9])[None]/299792458.)).astype(np.float32)
 if nf in [112,280,294]:
  crit=2.5+2*np.sinh(3*x[:,27]);phase=np.arange(8)/8;txh=x[:,31]
  extra=np.arcsinh((crit[:,None]-(3-np.remainder(txh[:,None]+phase,1)))/2)/3
  if nf in [280,294]:
   hh=torch.from_numpy(h).cuda();gg3=gg[:,:16].T.reshape(16,256,256)
   pv=profile_gpu(hh,tr,tc);sv=surface_gpu(hh,surface_fields(h),[tx_visibility(gg3,ht) for ht in [3,2]],tr,tc)
   x=np.concatenate((x,pv.cpu().numpy(),sv.cpu().numpy(),extra.astype(np.float32)),1)
  else:x=np.concatenate((x,extra.astype(np.float32)),1)
 if nf==294:
  xxg=torch.from_numpy(x).cuda();x=torch.cat((xxg,phase_features_gpu(xxg)),1)
 e3=(2*np.sinh(3*gg[:,:8])).max(1);e2=2*np.sinh(3*gg[:,8]);cos=gg[:,11]
 zone=np.select([(e3<0)&(e2>=0),(e2<0)&(e2>=-.5),(e2<-.5)&(abs(cos)<=.03),(e2<-.5)&(abs(cos)>.03),(e3>=0)&(e3<=2),e3>2],np.arange(6),default=255).astype(np.uint8)
 assert (zone<6).all()
 return x,fs,zone
arr=[np.lib.format.open_memmap(out/f'pl{b}.npy',mode='w+',dtype=np.float32,shape=(len(ids),256,256)) for b in ['415','58']]
zones=np.lib.format.open_memmap(out/'geometry_zones.npy',mode='w+',dtype=np.uint8,shape=(len(ids),256,256)) if a.save_zones else None
start=time.time();times=[]
for j,i in enumerate(ids):
 t=time.perf_counter();x,fs,zone=inputs(int(i));xxgpu=x if isinstance(x,torch.Tensor) else torch.from_numpy(x).cuda();pr=predict(xxgpu,fs);assert np.isfinite(pr).all()
 for k in [0,1]:arr[k][j]=pr[:,k].reshape(256,256)
 if zones is not None:zones[j]=zone.reshape(256,256)
 times.append(time.perf_counter()-t)
 if j%64==0:print(json.dumps({'done':j+1,'total':len(ids),'seconds':time.time()-start,'last_map_seconds':times[-1]}),flush=True)
for v in arr:v.flush()
if zones is not None:zones.flush()
meta={'run':a.run,'split':a.split,'part':a.part,'maps_per_band':len(ids),'precision':'BF16 network, FP32 numerical embeddings/readout/export/free-space','no_target_or_validity_mask_read':True,'input_fields':('Heightmap and TX only; all features recomputed on demand' if a.fresh else 'Heightmap, TX and input-only geometry caches'),'GPU_used':info,'GPU_memory_fraction_cap':.15,'GPU_peak_GB':torch.cuda.max_memory_allocated()/1e9,'fresh_geometry':a.fresh,'input_only_geometry_zones_saved':a.save_zones,'threads':a.threads,'seconds':time.time()-start,'cached_feature_map_seconds_mean':float(np.mean(times)),'cached_feature_map_seconds_p95':float(np.percentile(times,95)),'warm_map_seconds_mean':float(np.mean(times[1:])) if len(times)>1 else None,'warm_map_seconds_p95':float(np.percentile(times[1:],95)) if len(times)>1 else None,'cold_first_map_seconds':times[0],'latency_warning':('Includes constructing geometry on demand; CPU context, GPU trees and output. Shared GPU; official hardware protocol unconfirmed.' if a.fresh else 'Excludes constructing C3/C4 geometry caches. Not end-to-end official-runtime compliance.'),'checkpoint':'best.pt','selected_step':ck['step'],'shard':a.shard,'shards':a.shards}
(out/'inference.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

