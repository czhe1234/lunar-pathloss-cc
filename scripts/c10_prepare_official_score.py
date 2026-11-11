"""Adapt unchanged full VAL arrays to the public organizer scorer's file interface."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,hashlib,time,urllib.request,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
O=R/'audit/c10_official_scorer_20261101';O.mkdir(exist_ok=True)
assert not (O/'provenance.json').exists() and not list((O/'predictions').glob('*.npy'))
payload=json.loads((O/'source_payload.json').read_text())
S=O/'official';S.mkdir(exist_ok=True);P=O/'predictions';P.mkdir(exist_ok=True);G=O/'ground_truth/test';G.mkdir(parents=True,exist_ok=True)
REV='fa3ef9b89f964556753a18be102a5aadeb723377'
EXPECTED={'score_submission.py':'f9a96f5ef9c24e3d6dd8ebe6a6fccead25b9fc23','metrics.py':'d93af4591298bff8432a92dd2606d3e28bdf8062','LICENSE':'dba9f21fb8ebf5a241ab7271cfd1336bac604112'}
sources={}
for name,blob in EXPECTED.items():
 url=f'https://raw.githubusercontent.com/anderspearson206/lunar-radiomap-challenge/{REV}/{name}'
 body=payload[name].encode('utf-8')
 actual=hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest();assert actual==blob,(name,actual,blob)
 (S/name).write_bytes(body);sources[name]={'url':url,'git_blob':actual,'sha256':hashlib.sha256(body).hexdigest()}
src=R/'runs/c10_full_refit_s20261101/predictions/val_full_gpu_c10_fast_fresh'
mapping=json.loads((src/'indices.json').read_text());rows=list(csv.DictReader((D/'val_index.csv').open()))
assert mapping['source_split']=='val' and mapping['index_rows']==list(range(len(rows))) and len(rows)==2330
assert len({r['sample_id'] for r in rows})==len(rows)
pred={b:np.load(src/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']}
target={b:np.load(D/f'val_pl{b}.npy',mmap_mode='r') for b in pred}
mask={b:np.load(D/f'val_mask{b}.npy',mmap_mode='r') for b in pred}
idx=[];mix=[];valid={b:0 for b in pred};started=time.time()
for i,row in enumerate(rows):
 sid=row['sample_id'];ir={'sample_id':sid};mr={'sample_id':sid}
 for b in pred:
  pn=f'pl{b}_{sid}.npy';mn=f'mask{b}_{sid}.npy'
  p=np.asarray(pred[b][i]);t=np.asarray(target[b][i]);m=np.unpackbits(mask[b][i],bitorder='big').reshape(256,256)
  assert p.shape==t.shape==m.shape==(256,256) and p.dtype==np.float32 and np.isfinite(p).all()
  np.save(P/pn,p,allow_pickle=False);np.save(G/pn,t,allow_pickle=False);np.save(G/mn,m,allow_pickle=False)
  valid[b]+=int(m.sum());ir['pl'+b]=pn;mr['mask'+b]=mn
 idx.append(ir);mix.append(mr)
 if i%512==0:print(json.dumps({'maps_exported':i+1,'total':len(rows),'seconds':time.time()-started}),flush=True)
for name,records in [('index.csv',idx),('mask_index.csv',mix)]:
 with (G/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
assert valid=={'415':151537682,'58':135926253}
shutil.copy2(D/'metadata.json',O/'metadata.json')
manifest={'official_repository_revision':REV,'sources':sources,'dataset_split':'official fixed VAL, not withheld test','adapter_note':'The unmodified organizer scorer requires a directory named test. It contains all 2330 original VAL maps, without relabeling their evaluation scope. Only file layout and packed-mask decoding changed.','maps_per_band':len(rows),'valid_pixels':valid,'prediction_source':str(src),'original_targets_masks_and_predictions_changed':False,'scorer_source_modified':False,'source_acquisition':'Official GitHub connector; contents transferred and verified against pinned Git blob hashes','GPU_used':False,'expected_reference':json.loads((src/'independent_score.json').read_text())}
(O/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'prepared':str(O),'maps_per_band':len(rows),'valid_pixels':valid,'seconds':time.time()-started}),flush=True)
