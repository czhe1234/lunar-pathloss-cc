from pathlib import Path
import json,time,subprocess
R=Path("/home/chenzhe/work/lunar-pathloss-rm-challenge/CC")
A=R/"audit/c23_conditioning_20261101"
start=time.time()
while not (R/"runs/c22_fit128_rewarm/complete.json").exists():
 if time.time()-start>7200:raise RuntimeError("C22 predecessor did not finish within watchdog window")
 time.sleep(10)
assert json.loads((A/"interface_check.json").read_text())["zero_initialization_equivalence_passed"]
time.sleep(2)
commands=[
 ["./run_gpu.sh","4","src/c22_evaluate.py","--run","c22_fit128_rewarm","--checkpoint","last.pt","--part","fit","--dest","fit_last"],
 ["./run_gpu.sh","4","src/c23_small.py"],
 ["./run_gpu.sh","4","src/c22_evaluate.py","--run","c23_fit128_pair","--checkpoint","last_constant.pt","--part","fit","--dest","fit_constant","--condition","false"],
 ["./run_gpu.sh","4","src/c22_evaluate.py","--run","c23_fit128_pair","--checkpoint","last_conditional.pt","--part","fit","--dest","fit_conditional","--condition","true"],
]
for cmd in commands:
 for attempt in range(12):
  result=subprocess.run(cmd,cwd=R)
  if result.returncode!=75:break
  time.sleep(5)
 if result.returncode:raise RuntimeError((cmd,result.returncode))
constant=json.loads((R/"runs/c23_fit128_pair/fit_constant/metrics.json").read_text())
conditional=json.loads((R/"runs/c23_fit128_pair/fit_conditional/metrics.json").read_text())
delta=[b-a for a,b in zip(constant["rmse"],conditional["rmse"])]
out={"complete_fit_constant":constant,"complete_fit_conditional":conditional,"conditional_minus_constant_db":delta,"full_data_gate_passed":all(x<=-.05 for x in delta)}
(A/"small_gate.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out),flush=True)
