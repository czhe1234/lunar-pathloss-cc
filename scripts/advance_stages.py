"""Bounded continuation of the approved B1 -> B2 -> fixed official refit plan.
Every stage is sequential on the UUID-restricted GPU. Any failure stops the queue.
"""
import os,sys,json,time,fcntl,subprocess,traceback,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
p=argparse.ArgumentParser();p.add_argument('--after-run',default='b1_geometry_s20261031');p.add_argument('--after-pid',required=True,type=int);p.add_argument('--official-refit',action='store_true');args=p.parse_args()
lock=(ROOT/'.stage_controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state_path=ROOT/'logs/stage_controller.json'

def save(state,**kw):
 data={'state':state,'pid':os.getpid(),'updated_unix':time.time(),**kw};tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(state_path);print(json.dumps(data,ensure_ascii=False),flush=True)
def load_result(run):
 d=ROOT/'runs'/run
 if not (d/'progress.json').exists():raise RuntimeError(f'Upstream chain stopped before {run}; inspect logs/baseline_to_geometry.log and logs/geometry_sanity.log')
 assert json.loads((d/'progress.json').read_text())['status']=='completed'
 return json.loads((d/'result.json').read_text())
def launch(name,cmd,gpu=True):
 save('running_stage',stage=name,command=cmd)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','XDG_CACHE_HOME':str(ROOT/'.cache'),'MPLCONFIGDIR':str(ROOT/'.cache/matplotlib')})
 full=([str(ROOT/'run_gpu3.sh')] if gpu else [str(ROOT/'.venv/bin/python')])+cmd
 with (ROOT/'logs'/f'{name}.log').open('w') as log:
  child=subprocess.Popen(full,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (ROOT/'logs'/f'{name}.pid').write_text(str(child.pid)+'\n');save('running_stage',stage=name,child_pid=child.pid,command=cmd)
  rc=child.wait()
 if rc:raise RuntimeError(f'Stage {name} failed with exit code {rc}; see logs/{name}.log')
def verify_dev(run):
 path=ROOT/'runs'/run/'predictions/train_dev_bf16'
 if not (path/'inference.json').exists():launch(run+'_inference',['scripts/predict.py','--run',run,'--split','train','--part','dev'])
 if not (path/'independent_score.json').exists():launch(run+'_score',['scripts/score_predictions.py','--predictions',str(path.relative_to(ROOT))],False)
 s=json.loads((path/'independent_score.json').read_text());assert s['all_verification_checks_passed']
 latency=json.loads((path/'inference.json').read_text());assert latency['single_request_warm_ms']['mean']<500,latency
 launch(run+'_error_maps',['scripts/inspect_predictions.py','--predictions',str(path.relative_to(ROOT))],False)
 return s

def refresh():
 launch('stage_report',['scripts/report_progress.py'],False);launch('stage_plot',['scripts/plot_progress.py'],False)
try:
 save('waiting_for_b1',run=args.after_run,after_pid=args.after_pid)
 while True:
  try:os.kill(args.after_pid,0)
  except ProcessLookupError:break
  time.sleep(5)
 b0=load_result('b0_raw_s20261031');b1=load_result(args.after_run);verify_dev(args.after_run);refresh()
 # Do not hide a band trade-off inside a self-defined scalar score.
 if not all(b1[k]<=b0[k] for k in ('rmse415','rmse58')):
  save('needs_analysis',reason='B1 did not improve both bands relative to B0; inspect errors before expanding.',b0=b0,b1=b1);sys.exit(0)
 meta=ROOT/'features/train_profile.json'
 if not meta.exists():launch('build_train_profiles',['scripts/build_profiles.py','--split','train','--samples','192','--batch','8'])
 assert json.loads(meta.read_text())['samples_on_segment']==192
 b2name='b2_profile_s20261031'
 if not (ROOT/'runs'/b2name/'result.json').exists():launch(b2name,['scripts/train.py','--run',b2name,'--mode','profile','--batch','32','--epochs','60','--patience','10','--seed','20261031','--compile'])
 b2=load_result(b2name);verify_dev(b2name);refresh()
 if all(b2[k]<=b1[k] for k in ('rmse415','rmse58')):selected=b2name;selected_result=b2
 elif all(b1[k]<=b2[k] for k in ('rmse415','rmse58')):selected=args.after_run;selected_result=b1
 else:
  save('needs_analysis',reason='B1/B2 have a band trade-off. No automatic winner selection.',b1=b1,b2=b2);sys.exit(0)
 decision={'selected_dev_run':selected,'selected_dev_result':selected_result,'selection':'Both bands no worse than previous candidate, same split/seed/architecture/training limit. One development seed; no broad robustness claim.','profile_sampling':192,'second_seed_status':'Deferred while pursuing method gains; do not claim a two-seed confirmation.','next':'One fixed full-training refit for official stage confirmation, if enabled. This does not imply 1 dB will be met.'}
 (ROOT/'results/stage_selection.json').write_text(json.dumps(decision,indent=2)+'\n')
 if not args.official_refit:
  save('development_stages_completed',**decision);sys.exit(0)
 full='full_'+selected
 launch(full,['scripts/train_full.py','--run',full,'--selected-dev-run',selected])
 load_result(full)
 launch(full+'_official_inference',['scripts/predict.py','--run',full,'--split','val','--part','full'])
 pred=f'runs/{full}/predictions/val_full_bf16'
 launch(full+'_official_score',['scripts/score_predictions.py','--predictions',pred],False)
 score=json.loads((ROOT/pred/'independent_score.json').read_text())
 launch(full+'_official_error_maps',['scripts/inspect_predictions.py','--predictions',pred],False)
 refresh()
 save('goal_met' if score['goal_met_on_official_validation'] else 'official_stage_completed_needs_improvement',selected_run=full,official_rmse415_db=score['bands']['415']['rmse_db'],official_rmse58_db=score['bands']['58']['rmse_db'],goal_met=score['goal_met_on_official_validation'],note='No further blind retraining is queued. Inspect the saved errors and choose the next method change if the goal remains unmet.')
except BaseException as exc:
 if isinstance(exc,SystemExit) and exc.code in (0,None):raise
 save('failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raise
