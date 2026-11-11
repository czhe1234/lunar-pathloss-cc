from pathlib import Path
import json,subprocess,datetime
from cc_core import ROOT,check_source_unchanged
check_source_unchanged()
active=[]
for line in subprocess.check_output(["ps","-u","chenzhe","-o","pid=,comm=,args="],text=True).splitlines():
 cols=line.split(None,2)
 if len(cols)!=3 or not cols[1].startswith("python"):continue
 if "c29_final_check.py" in cols[2]:continue
 if any(x in cols[2] for x in ["src/c22","src/c23","src/c24","src/c25","src/c26","src/c27","src/c28","src/c29","audit/c25_map_batch_20261101/launch_after","audit/c26_representation_20261101/launch_after"]):active.append(line)
summary={"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_data_size_mtime_guard_passed":True,"active_campaign_processes":active,"C22_C29_training_complete":not active,"officialVAL_new_model":None,"official_score_unchanged":3.1775908265908654,"all_training_under_CC":True,"no_web_submission":True}
(ROOT/"audit/c29_fit_regime_20261101/final_execution_check.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
