from pathlib import Path
import subprocess,time,json
R=Path("/home/chenzhe/work/lunar-pathloss-rm-challenge/CC")
A=R/"audit/c22_scale_diagnosis_20261101"
start=time.time()
while not (A/"late_averaging_complete.json").exists():
 if time.time()-start>7200:raise RuntimeError("Missing averaged checkpoints")
 time.sleep(10)
time.sleep(3)
jobs=[]
for card,name in [(1,"c22_full_low"),(3,"c22_full_rewarm")]:
 log=(A/(name+"_post.log")).open("w")
 p=subprocess.Popen(["./run_gpu.sh",str(card),"src/c22_post.py","--run",name],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((p,card,name));log.close()
(A/"post_jobs.json").write_text(json.dumps([{"pid":p.pid,"gpu":c,"run":n} for p,c,n in jobs])+"\n")
for p,c,n in jobs:
 code=p.wait()
 if code:raise RuntimeError((n,code))
