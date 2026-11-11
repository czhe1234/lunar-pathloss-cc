from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[2];O=R/'audit/c38_density_continuation_20261101'
while not (R/'runs/c37_full_refit/predictions/val_full_c34_fresh_shard0of1/inference.json').exists():time.sleep(3)
with (O/'c37_package_same_gpu.log').open('w') as f:
 p=subprocess.run(['./run_gpu.sh','4','scripts/c34_verify.py'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
(O/'c37_package_same_gpu_exit.json').write_text(json.dumps({'exitcode':p.returncode})+'\n')
