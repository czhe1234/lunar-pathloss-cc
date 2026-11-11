"""Read-only numerical/source verification of public examples, CPU only."""
from pathlib import Path
import json,hashlib,datetime,csv
import numpy as np
R=Path(__file__).resolve().parents[1]; D=R.parent/'data'; O=R/'audit/generation_alignment_20261101'
checks=[]
def check(name,passed,detail=None):
 checks.append(dict(name=name,passed=bool(passed),detail=detail))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
for r in json.loads((O/'official_source_manifest.json').read_text()):
 p=R/r['file']; b=p.read_bytes()
 check('source SHA256 '+p.name,hashlib.sha256(b).hexdigest()==r['sha256'])
 check('source git blob '+p.name,hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==r['git_blob_sha1'])
for p in (O/'official_sources').glob('*.source.json'):
 m=json.loads(p.read_text());check('pinned Sionna source '+p.name,sha(R/m['file'])==m['sha256'])
ks=json.loads((O/'kaggle_official_files.json').read_text())
check('Kaggle pagination complete',ks['file_count_returned']==19 and not ks['pagination']['hasNextPageToken'],{'file_count':ks['file_count_returned'],'hasNextPageToken':ks['pagination']['hasNextPageToken']})
for r in json.loads((O/'official_local_file_sizes.json').read_text()):
 check('Kaggle/local size '+r['name'],(D/r['name']).stat().st_size==r['official_bytes'])
initial=json.loads((O/'server_file_inventory.json').read_text())
rawfiles=[r for r in initial['files'] if r['path'].startswith(str(D)+'/')]
for r in rawfiles:
 p=Path(r['path']);s=p.lstat()
 check('source size/mtime unchanged '+str(p.relative_to(D)),s.st_size==r['size'] and s.st_mtime_ns==r['mtime_ns'])
expected='4355f5e831aa280bf502c01488f823d67eee389e769e8d616644fa9d9a9ca907'
p=R/'runs/c4_full_refit_s20261031/best.pt';actual=sha(p)
check('C4 weight SHA256 unchanged',actual==expected,actual)
matched=json.loads((O/'public_example_release_matches.json').read_text());marker_records=[]
for r in matched:
 sid=r['public_sample'];x=np.load(O/f'public_examples/sample_{sid}_gt_415.npz');y=np.load(O/f'public_examples/sample_{sid}_gt_58.npz')
 pts=x['points'];xy=pts[:,:2].astype(int)
 check('sample '+sid+' all 65536 grid coordinates unique',len(np.unique(xy,axis=0))==65536)
 check('sample '+sid+' both bands use same height and coordinates',np.array_equal(x['height_yx'],y['height_yx']) and np.array_equal(pts,y['points']))
 check('sample '+sid+' visualization quadrilaterals only',x['quad_connectivity'].shape==(65025,4))
 best=r['best_height_matches'][0];match=r['tx_coordinate_matches'][0]
 check('sample '+sid+' identity orientation and near-exact height',best['public_to_release_d4']==0 and best['height_max_abs']<2e-6)
 txr,txc=int(match['tx_row']),int(match['tx_col']);z=x['height_yx'][txr,txc];mp=x['display_marker_points']
 marker_records.append({'public_sample':sid,'release_sample':match['sample_id'],'row':txr,'col':txc,'height_m':float(z),'marker_z_min_minus_H':float(mp[:,2].min()-z),'marker_z_max_minus_H':float(mp[:,2].max()-z),'display_cone_midpoint_minus_H':float((mp[:,2].min()+mp[:,2].max())/2-z),'interpretation':'Visualization cone, not saved transmitter position; do not infer actual TX z.'})
 check('sample '+sid+' both-band PL agreement below 0.03 dB',all(v['valid_rmse']<.03 for v in r['label_comparison'].values()))
# Existing prediction versus both versions of the same label. No model forward pass.
r=next(r for r in matched if r['public_sample']=='650');idx=r['tx_coordinate_matches'][0]['index_row'];predroot=R/'runs/c4_full_refit_s20261031/predictions/val_full_bf16'
# The exporter explicitly writes predictions in index order; check the saved export indices.
ids=json.loads((predroot/'indices.json').read_text())
assert ids['source_split']=='val' and ids['index_rows']==list(range(2330))
released_rows=list(csv.DictReader((D/'val_index.csv').open()))
assert ids['sample_ids']==[r['sample_id'] for r in released_rows]
assert ids['sample_ids'][idx]=='02136_00'
check('saved prediction indices and sample IDs match complete official validation',True)
score=[]
for band in ['415','58']:
 p=np.load(predroot/f'pl{band}.npy',mmap_mode='r')[idx].astype(float)
 y=np.load(D/f'val_pl{band}.npy',mmap_mode='r')[idx].astype(float)
 m=np.unpackbits(np.load(D/f'val_mask{band}.npy',mmap_mode='r')[idx],bitorder='big').reshape(256,256).astype(bool)
 public=30-np.load(O/f'public_examples/sample_650_gt_{band}.npz')['display_scalar_yx']
 rmse=lambda a,b:float(np.sqrt(np.mean((a[m]-b[m])**2)))
 a,b,c=rmse(p,y),rmse(p,public),rmse(y,public)
 check('sample 650 '+band+' RMSE reverse triangle inequality',abs(a-b)<=c+1e-12)
 score.append({'band':band,'sample_id':'02136_00','official_val_index_row':idx,'valid_pixels':int(m.sum()),'C4_vs_release_rmse_db':a,'C4_vs_public_float_grid_rmse_db':b,'release_vs_public_float_grid_rmse_db':c,'absolute_C4_rmse_change_db':abs(a-b),'scope':'one matched validation map, same release mask, saved frozen predictions only; not a new full-validation score'})
(O/'matched_sample_saved_prediction_check.json').write_text(json.dumps(score,indent=2)+'\n')
(O/'display_marker_audit.json').write_text(json.dumps(marker_records,indent=2)+'\n')
result={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training':False,'model_inference':False,'gpu_used':False,'simulation':False,'read_only_source_files_checked':len(rawfiles),'checks':checks,'passed':sum(x['passed'] for x in checks),'total':len(checks),'checkpoint_sha256':actual}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[r for r in checks if not r['passed']],'saved_prediction_check':score},indent=2))
assert result['passed']==result['total']
