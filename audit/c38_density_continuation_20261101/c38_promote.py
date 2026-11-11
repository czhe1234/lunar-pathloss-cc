"""Promote C38 only through complete DEV, then fixed full-TRAIN continuation and official VAL."""
from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[2];O=R/'audit/c38_density_continuation_20261101';jobs=[]
def waitfile(p):
 t=time.time()
 while not p.exists():
  if time.time()-t>10800:raise TimeoutError(str(p))
  time.sleep(3)
def launch(name,args):
 f=(O/(name+'.log')).open('w')
 p=subprocess.Popen(args,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append({'name':name,'pid':p.pid,'args':args});(O/'promotion_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');return p
def finish(p,name):
 ret=p.wait()
 if ret:raise RuntimeError((name,ret))
waitfile(R/'runs/c38_1024_continue/result.json')
res=json.loads((R/'runs/c38_1024_continue/result.json').read_text())
if not all(x<y-.01 for x,y in zip(res['DEV_RMSE'],[2.689824567907934,3.0420453947874186])):
 (O/'c38_promotion.json').write_text(json.dumps({'promoted':False,'stage':'sampledDEV','result':res})+'\n');raise SystemExit(0)
waitfile(O/'c37_package_same_gpu_exit.json')
time.sleep(5)
procs=[]
for k,card in enumerate([1,3,4]):
 procs.append(launch(f'c38_DEV_{k}',['./run_gpu.sh',str(card),'scripts/c34_predict.py','--run','c38_1024_continue','--split','train','--part','dev','--fresh','--shards','3','--shard',str(k)]))
for k,p in enumerate(procs):finish(p,f'C38 DEV shard{k}')
p=launch('c38_DEV_score',['.venv/bin/python','scripts/c34_score.py','--run','c38_1024_continue','--split','train','--shards','3']);finish(p,'C38 DEV score')
score=json.loads((R/'runs/c38_1024_continue/score_train_complete.json').read_text())
if not all(x<y-.01 for x,y in zip(score['RMSE'],[2.690645907435645,3.041642312198183])):
 (O/'c38_promotion.json').write_text(json.dumps({'promoted':False,'stage':'completeDEV','result':score})+'\n');raise SystemExit(0)
(O/'c38_promotion.json').write_text(json.dumps({'promoted':True,'stage':'completeDEV','result':score,'rule':'both bands at least0.01dB lower than C37, no official VAL selection'},indent=2)+'\n')
waitfile(R/'features/c38_dev_full/complete.json');time.sleep(3)
p=launch('c38_full_refit',['./run_gpu.sh','3','scripts/c38_refit.py','--parent','c38_1024_continue','--name','c38_full_refit']);finish(p,'C38 full continuation')
time.sleep(5);procs=[]
for k,card in enumerate([1,3,4]):
 procs.append(launch(f'c38_VAL_{k}',['./run_gpu.sh',str(card),'scripts/c34_predict.py','--run','c38_full_refit','--split','val','--part','full','--fresh','--shards','3','--shard',str(k)]))
for k,p in enumerate(procs):finish(p,f'C38 VAL shard{k}')
p=launch('c38_VAL_score',['.venv/bin/python','scripts/c34_score.py','--run','c38_full_refit','--split','val','--shards','3']);finish(p,'C38 VAL score')
p=launch('c38_official',['.venv/bin/python','scripts/c34_official.py','--run','c38_full_refit','--shards','3']);finish(p,'C38 official score')
(O/'c38_pipeline_complete.json').write_text(json.dumps({'complete':True,'official':'audit/c38_full_refit_official/verified_summary.json'})+'\n')
