"""Finish complete scoring after all raw-H/TX shards, then build the verified artifacts."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];run=R/'runs/c82_selected';shards=list(range(3))
for _ in range(2400):
    if all((run/f'predictions/val_full_c34_fresh_shard{s}of3/inference.json').is_file() for s in shards):break
    time.sleep(2)
else:raise RuntimeError('C82 prediction incomplete')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for tag,args,marker in [
    ('independent',['scripts/c34_score.py','--run','c82_selected','--split','val','--shards','3'],run/'score_val_complete.json'),
    ('official',['scripts/c34_official.py','--run','c82_selected','--shards','3'],R/'audit/c82_selected_official/verified_summary.json'),
    ('release',['scripts/c82_deliver.py'],R/'audit/c79_receiver/official_delivery.json')]:
    if marker.is_file():continue
    with (R/f'logs/c82_{tag}_score.log').open('w') as f:subprocess.run(['.venv/bin/python',*args],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print(json.load(open(R/'audit/c79_receiver/official_delivery.json'))['official']['RMSE'],flush=True)
