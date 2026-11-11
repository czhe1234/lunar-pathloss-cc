from pathlib import Path
import json,time,subprocess
R=Path("/home/chenzhe/work/lunar-pathloss-rm-challenge/CC")
A=R/"audit/c25_map_batch_20261101"
paths={k:R/"runs"/("c24_fit128_"+k)/"complete.json" for k in ["standard","wide","standard_headlr"]}
start=time.time()
while not all(p.exists() for p in paths.values()):
 if time.time()-start>7200:raise RuntimeError("C24 completion missing")
 time.sleep(10)
m={k:json.loads(p.read_text())["full_FIT"]["rmse"] for k,p in paths.items()}
capacity_passed=all(m["wide"][b]<=min(m["standard"][b],m["standard_headlr"][b])-.1 for b in [0,1])
(A/"predecessor_gate.json").write_text(json.dumps({"C24_metrics":m,"capacity_passed":capacity_passed})+"\n")
if capacity_passed:raise SystemExit(0)
time.sleep(3)
jobs=[]
for card,name in [(1,"c25_batch32_scaled"),(3,"c25_batch64_scaled"),(4,"c25_batch64_unscaled")]:
 script=A/(name+".sh")
 script.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd "+str(R)+"\n"+
  "./run_gpu.sh "+str(card)+" src/c25_train.py --variant "+name+" --steps 16 --eval-every 16 --suffix _smoke\n"+
  "./run_gpu.sh "+str(card)+" src/c25_train.py --variant "+name+" --steps 6000 --eval-every 1000\n"+
  "./run_gpu.sh "+str(card)+" src/c22_evaluate.py --run "+name+" --checkpoint best.pt --part dev --dest dev_complete\n")
 log=(A/(name+".log")).open("w")
 proc=subprocess.Popen(["bash",str(script)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((proc,card,name));log.close()
(A/"jobs.json").write_text(json.dumps([{"pid":p.pid,"gpu":c,"run":n} for p,c,n in jobs])+"\n")
for p,c,n in jobs:
 code=p.wait()
 if code:raise RuntimeError((n,code))
(A/"controller_complete.json").write_text(json.dumps({"completed":True})+"\n")
