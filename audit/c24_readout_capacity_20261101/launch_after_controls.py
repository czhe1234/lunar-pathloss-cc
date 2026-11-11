from pathlib import Path
import subprocess,time,json
R=Path("/home/chenzhe/work/lunar-pathloss-rm-challenge/CC")
A=R/"audit/c24_readout_capacity_20261101"
start=time.time()
gate=R/"audit/c23_conditioning_20261101/small_gate.json"
posts=[R/"audit/c22_scale_diagnosis_20261101"/(n+"_post.json") for n in ["c22_full_low","c22_full_rewarm"]]
while not (gate.exists() and all(p.exists() for p in posts)):
 if time.time()-start>7200:raise RuntimeError("C22/C23 predecessor missing")
 time.sleep(10)
if json.loads(gate.read_text())["full_data_gate_passed"]:
 (A/"not_started.json").write_text(json.dumps({"reason":"C23 passed; capacity fallback not launched"})+"\n")
 raise SystemExit(0)
assert json.loads((A/"interface_check.json").read_text())["function_preservation_passed"]
time.sleep(3)
jobs=[]
for card,name in [(1,"standard"),(3,"wide"),(4,"standard_headlr")]:
 log=(A/(name+".log")).open("w")
 proc=subprocess.Popen(["./run_gpu.sh",str(card),"src/c24_train.py","--variant",name],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((proc,card,name));log.close()
(A/"jobs.json").write_text(json.dumps([{"pid":p.pid,"gpu":c,"variant":n} for p,c,n in jobs])+"\n")
for p,c,n in jobs:
 code=p.wait()
 if code:raise RuntimeError((n,code))
