import sys,os,json,time,subprocess,fcntl,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);fit='c4_critical_length_fast_s20261031';full='c4_full_refit_s20261031';state=ROOT/'audit/c4_delivery_progress.json'
def update(obj):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(state);print(json.dumps(obj),flush=True)
def run(stage,cmd):
 update({'stage':stage,'command':cmd,'time':time.time()})
 subprocess.run(cmd,cwd=ROOT,check=True)
try:
 lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 import torch
 fr=ROOT/'runs'/fit;ck=torch.load(fr/'best.pt',map_location='cpu',weights_only=False);epoch=ck['epoch'];del ck
 history=json.loads((fr/'history.json').read_text());row=next(a for a in history if a['epoch']==epoch);result={**row['dev'],'state':'development_branch_stopped','run':fit,'best_epoch':epoch,'epochs_completed':history[-1]['epoch'],'evaluation_split':'development','part':'fit','source':'Training evaluator on best checkpoint; independent input-only replay follows.'}
 (fr/'result.json').write_text(json.dumps(result,indent=2)+'\n');(fr/'frozen_selection.json').write_text(json.dumps({'checkpoint_epoch':epoch,'checkpoint_sha256':hashlib.sha256((fr/'best.pt').read_bytes()).hexdigest(),'development_scores':row['dev'],'official_validation_used':False},indent=2)+'\n')
 run('independent_development_predictions',['./run_gpu3.sh','scripts/c4_predict.py','--run',fit,'--split','train','--part','dev','--batch','8','--latency-samples','32'])
 run('independent_development_score',['.venv/bin/python','scripts/score_predictions.py','--predictions',f'runs/{fit}/predictions/train_dev_bf16'])
 target=ROOT/'runs'/full;target.mkdir(exist_ok=False)
 (target/'PLAN.md').write_text('Frozen before official validation: initialize from development best; all 25,720 training maps; 8 epochs; lr=0.0001; no validation checkpoint selection. Official validation is evaluated once after freezing the epoch-8 model.\n')
 run('fixed_full_training',['./run_gpu3.sh','scripts/c4_train_fast.py','--run',full,'--part','full','--epochs','8','--batch','32','--lr','0.0001','--compile','--init',fit])
 run('official_validation_predictions',['./run_gpu3.sh','scripts/c4_predict.py','--run',full,'--split','val','--part','full','--batch','8','--latency-samples','64'])
 run('independent_official_validation_score',['.venv/bin/python','scripts/score_predictions.py','--predictions',f'runs/{full}/predictions/val_full_bf16'])
 run('export_scored_version',['.venv/bin/python','scripts/export_lunar_version.py','--run',full])
 update({'stage':'exported','run':full,'time':time.time(),'requires_final_package_smoke_check':True})
except Exception as exc:
 update({'stage':'failed','error':repr(exc),'time':time.time()});traceback.print_exc();sys.exit(1)
