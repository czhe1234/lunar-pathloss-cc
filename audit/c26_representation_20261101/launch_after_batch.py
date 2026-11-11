from pathlib import Path
import subprocess,time,json,os
R=Path("/home/chenzhe/work/lunar-pathloss-rm-challenge/CC")
A=R/"audit/c26_representation_20261101"
start=time.time()
while not (R/"audit/c25_map_batch_20261101/controller_complete.json").exists():
 if time.time()-start>7200:raise RuntimeError("C25 completion missing")
 time.sleep(10)
time.sleep(3)
jobs=[]
for card,kind in [(1,"main"),(3,"subset_standard"),(4,"subset_wide")]:
 script=A/(kind+".sh")
 script.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd "+str(R)+"\n"+
  "./run_gpu.sh "+str(card)+" src/c26_features.py --kind "+kind+"\n"+
  "OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 .venv/bin/python src/c26_ridge.py --kind "+kind+"\n")
 log=(A/(kind+".log")).open("w")
 p=subprocess.Popen(["bash",str(script)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((p,card,kind));log.close()
(A/"jobs.json").write_text(json.dumps([{"pid":p.pid,"gpu":c,"kind":k} for p,c,k in jobs])+"\n")
for p,c,k in jobs:
 code=p.wait()
 if code:raise RuntimeError((k,code))
(A/"complete.json").write_text(json.dumps({"complete":True})+"\n")
