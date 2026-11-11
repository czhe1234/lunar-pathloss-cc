import time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];jobs={}
pending={3:'both',4:'patch'}
while pending:
 for card,kind in list(pending.items()):
  p=R/'runs'/('c48_'+kind+'_tiny')/'result.json'
  if not p.exists():continue
  result=json.loads(p.read_text())
  if not result['tiny_both_le_080db']:
   jobs[kind]={'skipped':'tiny_fit_gate_failed','FIT':result['FIT_probe']['RMSE']};del pending[card];continue
  name='c48_'+kind+'_full';f=open(R/('logs/'+name+'.log'),'w')
  args=['./run_gpu.sh',str(card),'src/c48_train.py','--scope','full','--kind',kind,'--name',name,'--batch','2048','--steps','4000']
  proc=subprocess.Popen(args,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  jobs[kind]={'pid':proc.pid,'card':card,'args':args};del pending[card]
  (R/'audit/c48_fresnel_corridor_20261102/full_launch.json').write_text(json.dumps(jobs,indent=2)+'\n')
 time.sleep(3)
print(jobs,flush=True)
