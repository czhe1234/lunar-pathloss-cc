"""Bounded C44 follow-up on an H100 once its C43 work and replay finish."""
from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[1];A=R/'audit/c44_dense_profile_20261102'
ready=R/'runs/c43_grid_fit256/fp32/fit_panel_full.json'
deadline=time.time()+2700
while not ready.exists():
    assert time.time()<deadline,'Timed out waiting for C43 GPU1'
    time.sleep(10)
time.sleep(2)
with (R/'logs/c44_probe.log').open('x') as f:
    subprocess.run(['./run_gpu.sh','1','src/c44_train.py','--variant','path','--stage','probe'],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
p=json.loads((A/'probe.json').read_text())
assert p['B_C_initial_weights_identical'] and p['coordinate_check']
b=p['benchmarks'][0]
assert b['batch']==16 and b['step_seconds']*4000<1800,('C44 screen exceeds predeclared30min training budget',b)
name='c44_profile_fit64'
cmd=['./run_gpu.sh','1','src/c44_train.py','--variant','path','--stage','fit64','--batch','16','--name',name]
with (R/'logs'/f'{name}.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT)
    (A/'launch64.json').write_text(json.dumps({'pid':child.pid,'card':1,'command':cmd},indent=2)+'\n')
    code=child.wait()
assert code==0,code
with (R/'logs'/f'{name}_fp32.log').open('x') as f:
    subprocess.run(['./run_gpu.sh','1','scripts/c44_fp32.py','--run',name],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
print('C44 screening and canonical replay complete',flush=True)
