from pathlib import Path
import subprocess,json,time
R=Path(__file__).resolve().parents[1];run=R/'runs/c44_profile_fit64'
deadline=time.time()+3000
while not (run/'result.json').exists():
 assert time.time()<deadline,'C44 training timeout; inspect saved progress before more compute'
 time.sleep(10)
time.sleep(2)
with (R/'logs/c44_profile_fit64_fp32.log').open('x') as f:
 subprocess.run(['./run_gpu.sh','3','scripts/c44_fp32.py','--run',run.name],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'logs/c44_compare64.log').open('x') as f:
 subprocess.run(['.venv/bin/python','scripts/c44_compare.py','--stage','fit64'],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
print('C44 matched screen verified',flush=True)
