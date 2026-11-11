"""CPU-only input representation experiment, fixed existing terrain split.
Baseline predictions are saved for evaluation only, never included in X.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json,csv,time,hashlib
import numpy as np
from scipy.ndimage import map_coordinates,uniform_filter
R=Path(__file__).resolve().parents[1];D=R.parent/'data';O=R/'features/c6_points';O.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((D/'train_index.csv').open()));dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
H=np.load(D/'train_hm.npy',mmap_mode='r');G=[np.load(R/f'features/{n}/train.npy',mmap_mode='r') for n in ['c3','c4_extra']]
Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];M=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
B=R/'runs/c4_critical_length_fast_s20261031/predictions/train_dev_bf16';bm=json.loads((B/'indices.json').read_text());bpos={i:j for j,i in enumerate(bm['index_rows'])};BP=[np.load(B/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
basic_names=['height','relative_height','highpass5','slope_x','slope_y','dx','dy','dz','log_distance','tx_pixel','tx_neighborhood']
geom_names=[f'profile_max_bin{k}' for k in range(8)]+['margin_tx2','underground_fraction','max_margin_location','incidence_cos','joint_visibility_near','joint_visibility_mid','joint_visibility_far','min_excess_distance','critical_TX_height','critical_RX_height','underground_length_TX3','underground_length_TX2']
names=basic_names+geom_names+[f'ordered_clearance_{k:03}' for k in range(128)];assert len(names)==159
@lru_cache(maxsize=96)
def terrain(i):
 h=np.array(H[i],dtype=np.float32);sy,sx=np.gradient(h);hp=h-uniform_filter(h,size=5,mode='nearest');return h,sy,sx,hp

def item(task):
 i,n=task;row=rows[i];h,sy,sx,hp=terrain(int(row['hm_row']));tr=int(row['tx_row']);tc=int(row['tx_col']);rng=np.random.default_rng(20261101+104729*i)
 pix=rng.choice(65536,n,replace=False);rr,cc=np.divmod(pix,256);dx=cc.astype(np.float32)-tc;dy=rr.astype(np.float32)-tr;ht=float(h[tr,tc]);dz=h[rr,cc]+1-ht-3;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
 basic=np.stack([h[rr,cc]/100,(h[rr,cc]-ht)/50,hp[rr,cc]/5,sx[rr,cc]/5,sy[rr,cc]/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((rr==tr)&(cc==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)],1)
 gg=np.concatenate([np.asarray(a[i,:,rr,cc],dtype=np.float32) for a in G],1);assert gg.shape==(n,20)
 t=(np.arange(128,dtype=np.float32)+.5)/128
 yy=tr+dy[:,None]*t;xx=tc+dx[:,None]*t
 hh=map_coordinates(h,np.stack((yy,xx)),order=1,mode='nearest',prefilter=False)
 e=hh-(ht+3+dz[:,None]*t);profile=np.arcsinh(e/2)/3
 x=np.concatenate((basic,gg,profile),1).astype(np.float32);fs=20*np.log10(4*np.pi*dist[:,None]*np.array([415e6,5.775e9])[None]/299792458.)
 target=np.stack([np.asarray(a[i,rr,cc],dtype=np.float32) for a in Y],1)
 valid=np.stack([np.unpackbits(a[i],bitorder='big').reshape(256,256)[rr,cc] for a in M],1)
 baseline=np.stack([a[bpos[i],rr,cc] for a in BP],1) if i in bpos else None
 return x,target,valid,fs.astype(np.float32),pix.astype(np.int32),baseline

selection={part:[i for i,r in enumerate(rows) if (r['terrain_id'] in dev)==(part=='dev')] for part in ['fit','dev']}
assert {rows[i]['terrain_id'] for i in selection['fit']}.isdisjoint({rows[i]['terrain_id'] for i in selection['dev']})
meta={'source_split':'official train','development_terrain_split':'audit/internal_split.json','features':names,'summary_channels':31,'ordered_profile_channels':128,'pixel_sampling':'uniform without replacement per map; independent of masks and labels; deterministic seed 20261101+104729*index_row','CPU_only':True,'baseline_predictions_enter_X':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parts':{}}
start=time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
 for part in ['fit','dev']:
  ids=selection[part];n=64 if part=='fit' else 256;N=len(ids)*n
  if (O/f'{part}_complete.json').exists():raise RuntimeError('Existing complete point dataset; preserve it')
  fields={'x':(np.float32,(N,159)),'y':(np.float32,(N,2)),'mask':(np.uint8,(N,2)),'fspl':(np.float32,(N,2)),'pixel':(np.int32,(N,))}
  if part=='dev':fields['baseline']=(np.float32,(N,2))
  out={k:np.lib.format.open_memmap(O/f'{part}_{k}.npy',mode='w+',dtype=dtype,shape=shape) for k,(dtype,shape) in fields.items()}
  for lo in range(0,len(ids),128):
   results=pool.map(item,[(i,n) for i in ids[lo:lo+128]])
   for j,(x,y,m,fs,pix,bp) in enumerate(results,lo):
    sl=slice(j*n,(j+1)*n);out['x'][sl]=x;out['y'][sl]=y;out['mask'][sl]=m;out['fspl'][sl]=fs;out['pixel'][sl]=pix
    if part=='dev':out['baseline'][sl]=bp
   if lo%2048==0:print(json.dumps({'part':part,'maps_done':min(lo+128,len(ids)),'maps_total':len(ids),'seconds':time.time()-start}),flush=True)
  for v in out.values():v.flush()
  np.save(O/f'{part}_map_indices.npy',np.array(ids,dtype=np.int64))
  rec={'maps':len(ids),'terrains':len({rows[i]['terrain_id'] for i in ids}),'pixels_per_map':n,'sampled_pixels':N,'valid_pixels_per_band':out['mask'].sum(0).astype(int).tolist(),'complete':True}
  meta['parts'][part]=rec;(O/f'{part}_complete.json').write_text(json.dumps(rec,indent=2)+'\n')
  print(json.dumps({'completed':part,**rec}),flush=True)
meta['seconds']=time.time()-start;(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'complete':True,'seconds':meta['seconds']}),flush=True)
