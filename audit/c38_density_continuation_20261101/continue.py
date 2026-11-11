"""Continue authorized C37 validation/refit and matched point-density controls."""
from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[2]
O=R/'audit/c38_density_continuation_20261101'
jobs=[]
def record():
 (O/'active_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
def waitfile(p):
 start=time.time()
 while not p.exists():
  if time.time()-start>7200:raise RuntimeError(f'timed out waiting for{p}')
  time.sleep(3)
def launch(name,args):
 f=(O/(name+'.log')).open('w')
 p=subprocess.Popen(args,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append({'name':name,'pid':p.pid,'args':args});record();return p
def finish(p,name):
 code=p.wait()
 if code:raise RuntimeError(f'{name} exited{code}')
for k in range(3):waitfile(R/f'runs/c37_wide_periodic8/predictions/train_dev_c34_fresh_shard{k}of3/inference.json')
time.sleep(3)
p=launch('c37_complete_DEV',['.venv/bin/python','scripts/c34_score.py','--run','c37_wide_periodic8'])
finish(p,'complete DEV score')
score=json.loads((R/'runs/c37_wide_periodic8/score_train_complete.json').read_text())
assert all(x<y for x,y in zip(score['RMSE'],[2.952572673204586,3.3930735134224634])),score
control=launch('c39_control',['./run_gpu.sh','3','src/c38_continue.py','--mode','periodic','--points','256','--steps','40000','--name','c39_256_continue'])
full=launch('c37_full_refit',['./run_gpu.sh','1','scripts/c34_refit.py','--parent','c37_wide_periodic8','--name','c37_full_refit'])
finish(full,'full TRAIN refit')
val=launch('c37_VAL',['./run_gpu.sh','4','scripts/c34_predict.py','--run','c37_full_refit','--split','val','--part','full','--fresh','--shards','1','--shard','0'])
waitfile(R/'features/c38_dense/complete.json')
dense=launch('c38_dense',['./run_gpu.sh','1','src/c38_continue.py','--mode','periodic','--points','1024','--steps','40000','--name','c38_1024_continue'])
finish(val,'official VAL prediction')
p=launch('c37_VAL_score',['.venv/bin/python','scripts/c34_score.py','--run','c37_full_refit','--split','val','--shards','1']);finish(p,'VAL independent score')
p=launch('c37_official',['.venv/bin/python','scripts/c34_official.py','--run','c37_full_refit','--shards','1']);finish(p,'official scorer')
finish(control,'256point control');finish(dense,'1024point candidate')
(O/'controller_complete.json').write_text(json.dumps({'C37_official_complete':True,'matched_continuations_complete':True},indent=2)+'\n')
