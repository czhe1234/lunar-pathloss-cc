from pathlib import Path
import numpy as np,json,csv,hashlib,datetime
R=Path(__file__).resolve().parents[1];A=R/'audit/c141_direct_receiver'
panel=json.loads((A/'fit_panel.json').read_text())['records'];protocol=json.loads((R/'audit/c136_native/protocol.json').read_text())
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
fitids={rows[i]['terrain_id'] for i in protocol['fit']};monids={rows[i]['terrain_id'] for i in protocol['monitor']}
selids={r['terrain_id'] for r in panel}
assert selids<=fitids and not(selids&monids)
errs=[];counts=[];C=(299792458/(4*np.pi*np.array([415e6,5.775e9])))**2
Y=[np.load(R.parent/f'data/train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/f'data/train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
strict=np.zeros(2,int)
for rec in panel:
 i=rec['map'];px=rec['pixels'];assert rows[i]['sample_id']==rec['sample_id']
 for b in range(2):
  y=np.asarray(Y[b][i]).ravel()[px].astype(float);want=np.array(rec['labels'])[:,b]
  errs.append(float(np.max(abs(y-want))))
  masks=np.unpackbits(M[b][i]).ravel()[px].astype(bool);assert np.all(masks);counts.append(len(px))
 z=np.load(A/f'fit_replay/{i}.npz');w=z['weights'];D=z['native_full_native'][...,None]*C
 allow=np.power(10.,-np.asarray(rec['labels'])/10.)
 wrong=np.all((D>allow[:,None,:]*10**.1)|(w[:,:,None]<=1e-12),axis=1)
 strict+=(wrong&np.asarray(rec['is_case'])[:,None]).sum(0)
assert max(errs)==0 and strict.tolist()==[22,22]
out={'checked_at':datetime.datetime.now().astimezone().isoformat(),'fit_terrains':len(selids),'monitor_overlap':len(selids&monids),
 'selected_maps':len(panel),'label_max_absolute_difference':max(errs),'valid_points_per_band':[sum(counts[::2]),sum(counts[1::2])],
 'independent_linear_power_strict_donor_counterexamples':strict.tolist(),'official_VAL_labels_read':False,
 'panel_sha256':hashlib.sha256((A/'fit_panel.json').read_bytes()).hexdigest(),
 'input_only_sha256':hashlib.sha256((A/'input_panel.json').read_bytes()).hexdigest()}
(A/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
