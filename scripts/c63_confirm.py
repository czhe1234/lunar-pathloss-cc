from pathlib import Path
import json,time,subprocess
import numpy as np
R=Path.cwd()
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 try:return p.read_text().split()[2]!='Z'
 except FileNotFoundError:return False
while alive(3767361):time.sleep(3)
meta=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1/inference.json'
if meta.exists() and json.loads(meta.read_text()).get('complete'):
 with open('logs/c54_fullDEV_score.log','w') as f:subprocess.run(['.venv/bin/python','scripts/c60_score.py','--run','c54_combo_dev','--scope','all'],stdout=f,stderr=subprocess.STDOUT,check=True)
while alive(3762461) or alive(3762462):time.sleep(3)
records={}
for name in ['c63_fixed_full','c63_dynamic_full']:
 run=R/'runs'/name
 if not (run/'result.json').exists():
  records[name]={'error':'process ended without result; inspect existing log'};continue
 result=json.loads((run/'result.json').read_text());cfg=json.loads((run/'config.json').read_text())
 pred=np.load(run/'dev_prediction.npy');y=np.load('features/c6_points/dev_y.npy',mmap_mode='r');m=np.load('features/c6_points/dev_mask.npy',mmap_mode='r')
 rm=[]
 for b in range(2):
  keep=m[:,b].astype(bool);e=pred[keep,b].astype(float)-y[keep,b].astype(float);rm.append(float(np.sqrt(np.mean(e*e))))
 assert max(abs(np.array(rm)-result['DEV']['RMSE']))<1e-8
 records[name]={'result':result,'independent_CPU_RMSE':rm,'config':cfg}
if all('config' in r for r in records.values()):
 assert records['c63_fixed_full']['config']['initial_state_sha256']==records['c63_dynamic_full']['config']['initial_state_sha256']
 hh=[json.loads((R/'runs'/n/'history.json').read_text()) for n in records]
 common=sorted(set(x['step'] for x in hh[0])&set(x['step'] for x in hh[1]))
 records['shared_steps']=[{'step':st,'fixed_DEV':next(x['DEV']['RMSE'] for x in hh[0] if x['step']==st),'dynamic_DEV':next(x['DEV']['RMSE'] for x in hh[1] if x['step']==st)} for st in common]
(R/'audit/c63_dynamic_full_20261102/comparison.json').write_text(json.dumps(records,indent=2)+'\n')
candidate=records.get('c63_dynamic_full',{}).get('result',{})
if candidate.get('both_gain015db'):
 jobs=[]
 for card,shard in [(1,0),(3,1)]:
  log=open('logs/c63_dynamic_dev64_shard'+str(shard)+'.log','w')
  p=subprocess.Popen(['./run_gpu.sh',str(card),'src/c63_predict.py','--run','c63_dynamic_full','--scope','dev64','--shard',str(shard),'--of','2'],stdout=log,stderr=subprocess.STDOUT)
  jobs.append(p)
 for p in jobs:assert p.wait()==0
 with open('logs/c63_dev64_score.log','w') as f:subprocess.run(['.venv/bin/python','scripts/c63_score.py','--run','c63_dynamic_full','--scope','dev64'],stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'audit/c63_dynamic_full_20261102/confirmation_controller_complete.json').write_text(json.dumps({'complete':True,'dynamic_DEV64_gate':bool(candidate.get('both_gain015db'))}))
