"""Launch authorized C33 jobs after their current CC GPU jobs finish."""
from pathlib import Path
import json,time,subprocess,importlib.util
R=Path(__file__).resolve().parents[2];A=R/"audit/c33_gpu_trees_20261101"
while importlib.util.find_spec("catboost") is None:
 time.sleep(5)
jobs=[]
for mode,card,band in [("cold",1,"415"),("warm",3,"58")]:
 script=A/("band"+band+".sh")
 script.write_text("set -eu\ncd "+str(R)+"\nwhile [ ! -f runs/c32_"+mode+"/complete.json ]; do sleep 5; done\nsleep 2\n./run_gpu.sh "+str(card)+" src/c33_train.py --band "+band+" --smoke\n./run_gpu.sh "+str(card)+" src/c33_train.py --band "+band+"\n")
 log=(A/("band"+band+".log")).open("w")
 proc=subprocess.Popen(["bash",str(script)],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
 jobs.append({"band":band,"card":card,"pid":proc.pid,"script":str(script)})
(A/"jobs.json").write_text(json.dumps(jobs,indent=2)+"\n")
print(json.dumps(jobs),flush=True)
