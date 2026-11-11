"""Input-only dense C10 representation cache for one bounded spatial residual trial."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0'
import sys,csv,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from c10_inference import C10Predictor
from cc_core import check_source_unchanged,gpu_guard
check_source_unchanged();info=gpu_guard()
O=R/'features/c17_spatial';O.mkdir(exist_ok=False);(O/'source').mkdir()
for p in [Path(__file__),R/'src/c10_inference.py',R/'audit/c17_protocol.json']:
 shutil.copy2(p,O/'source'/p.name)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
rng=np.random.default_rng(20261101);groups={}
for i,r in enumerate(rows):groups.setdefault(r['terrain_id'],[]).append(i)
ids={};terrains={}
for part,n in [('fit',512),('dev',128)]:
 pool=sorted(t for t in groups if (t in dev)==(part=='dev'))
 terrains[part]=rng.choice(pool,n,replace=False).tolist()
 ids[part]=[int(rng.choice(groups[t])) for t in terrains[part]]
assert set(terrains['fit']).isdisjoint(terrains['dev'])
model=C10Predictor(R/'runs/c10_points_phase_geometry_s20261101')
keep=list(range(31))+list(range(280,294))
class Tap:
 def __init__(self,forest):self.forest=forest;self.latest=None
 def predict(self,x):
  self.latest=x[:,keep].T.reshape(45,256,256).cpu().numpy()
  return self.forest.predict(x)
tap=Tap(model.forests[0]);model.forests[0]=tap
saved=R/'runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh'
mapping=json.loads((saved/'indices.json').read_text());oldmap={v:i for i,v in enumerate(mapping['index_rows'])}
old=[np.load(saved/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
metadata={'scope':'512 FIT and128 DEV terrain-disjoint maps, one random TX per terrain, selection uses only IDs',
 'ids':ids,'terrains':terrains,'seed':20261101,'GPU':info,'input_only':True,
 'features':'C10 basic31 + phase14 + predicted dual excess/20, 47 channels; all deterministic from heightmap/TX and frozen FIT-only C10',
 'C10_model_sha256':model.model_sha256,'official_VAL_used':False,'checks':[]}
(O/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
yy,xx=np.mgrid[:256,:256];yy=yy.astype(np.float32);xx=xx.astype(np.float32)
start=time.time()
for part in ['fit','dev']:
 x=np.lib.format.open_memmap(O/f'{part}_x.npy',mode='w+',dtype=np.float16,shape=(len(ids[part]),47,256,256))
 pred=np.lib.format.open_memmap(O/f'{part}_base.npy',mode='w+',dtype=np.float32,shape=(len(ids[part]),2,256,256))
 for j,i in enumerate(ids[part]):
  row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);tr=int(row['tx_row']);tc=int(row['tx_col'])
  p=model.predict(h,tr,tc)
  d=np.sqrt((xx-tc)**2+(yy-tr)**2+(h+1-float(h[tr,tc])-3)**2).clip(.25)
  fs=(20*np.log10(4*np.pi*d[None]*np.array([415e6,5.775e9])[:,None,None]/299792458.)).astype(np.float32)
  v=np.concatenate((tap.latest,(p-fs)/20),0)
  assert np.isfinite(v).all()
  x[j]=v;pred[j]=p
  if part=='dev' and j<4:
   diff=max(float(np.max(abs(p[k]-old[k][oldmap[i]]))) for k in [0,1])
   assert diff<1e-4,(i,diff)
   metadata['checks'].append({'dev_map':i,'max_saved_C10_difference_db':diff,'feature_FP16_max_error':float(np.max(abs(v-x[j].astype(np.float32))))})
  if j%64==0:print(json.dumps({'part':part,'done':j+1,'total':len(ids[part]),'seconds':time.time()-start}),flush=True)
 x.flush();pred.flush()
metadata.update({'state':'completed','seconds':time.time()-start,'peak_GPU_GB':torch.cuda.max_memory_allocated()/1e9})
check_source_unchanged();(O/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({k:v for k,v in metadata.items() if k not in ['ids','terrains']}),flush=True)
