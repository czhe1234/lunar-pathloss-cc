from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[2];O=R/'audit/c38_density_continuation_20261101'
def waitfile(p):
 start=time.time()
 while not p.exists():
  if time.time()-start>7200:raise TimeoutError(str(p))
  time.sleep(3)
def run(name,args):
 with (O/(name+'.log')).open('w') as f:
  p=subprocess.run(args,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError((name,p.returncode))
waitfile(R/'runs/c37_full_refit/predictions/val_full_c34_fresh_shard0of1/inference.json')
if not (R/'runs/c37_full_refit/score_val_complete.json').exists():
 run('c37_VAL_score',['.venv/bin/python','scripts/c34_score.py','--run','c37_full_refit','--split','val','--shards','1'])
if not (R/'audit/c37_full_refit_official/verified_summary.json').exists():
 run('c37_official',['.venv/bin/python','scripts/c34_official.py','--run','c37_full_refit','--shards','1'])
waitfile(R/'runs/c38_1024_continue/result.json')
(O/'controller_complete.json').write_text(json.dumps({'C37_official_complete':True,'matched_continuations_complete':True})+'\n')
