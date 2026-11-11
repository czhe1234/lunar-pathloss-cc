"""Run the previously verified unmodified organizer scorer on new VAL predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,argparse,csv,json,subprocess
import numpy as np
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shards',type=int,default=3);a=p.parse_args()
run=R/'runs'/a.run
ref=json.loads((run/'score_val_complete.json').read_text())
assert ref['maps']==2330 and ref['complete_requested_split']
shared=R/'audit/c10_official_scorer_20261101'
o=R/'audit'/f'{a.run}_official';o.mkdir(exist_ok=False);predout=o/'predictions';predout.mkdir()
rows=list(csv.DictReader((R.parent/'data/val_index.csv').open()));seen=[]
for shard in range(a.shards):
 d=run/'predictions'/f'val_full_c34_fresh_shard{shard}of{a.shards}'
 assert (d/'inference.json').exists()
 ids=json.loads((d/'indices.json').read_text())['index_rows'];seen+=ids
 arrays=[np.load(d/f'pl{b}.npy',mmap_mode='r') for b in ['415','58']]
 for j,i in enumerate(ids):
  for k,b in enumerate(['415','58']):
   np.save(predout/f"pl{b}_{rows[i]['sample_id']}.npy",arrays[k][j],allow_pickle=False)
assert len(seen)==2330 and set(seen)==set(range(2330))
cmd=[str(R/'.venv/bin/python'),str(shared/'official/score_submission.py'),'--submission',str(predout),'--gt-root',str(shared/'ground_truth'),'--data-root',str(shared/'ground_truth'),'--meta',str(shared/'metadata.json'),'--band','both','--out',str(o/'official_report.json')]
with (o/'official.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
off=json.loads((o/'official_report.json').read_text())
vals=[off['per_band'][b]['rmse_db_masked'] for b in ['415','58']]
assert max(abs(x-y) for x,y in zip(vals,ref['RMSE']))<1e-5
counts=[sum(int(round(r['valid_frac']*65536)) for r in off['per_sample'] if r['band']==b) for b in ['415','58']]
assert counts==ref['valid_pixels']==[151537682,135926253]
out={'model':a.run,'scope':'complete official fixed validation,2330maps/band; not hidden test','RMSE':vals,'combined_rmse_db_masked':off['summary']['combined_rmse_db_masked'],'valid_pixels':counts,'goal_both_le_1db':all(v<=1 for v in vals),'C10_official_RMSE':[2.96720214449994,3.3879795086817914],'relative_gain_vs_C10':(1-np.array(vals)/np.array([2.96720214449994,3.3879795086817914])).tolist(),'official_source_revision':'fa3ef9b89f964556753a18be102a5aadeb723377','scorer':'Reused previously verified unmodified official score_submission.py and metrics.py','source_and_groundtruth_adapter':str(shared),'official_report':str(o/'official_report.json'),'independent_score':str(run/'score_val_complete.json'),'website_submission':False}
(o/'verified_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
