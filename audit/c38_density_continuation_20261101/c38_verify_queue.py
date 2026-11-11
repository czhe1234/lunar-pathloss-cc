from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[2];O=R/'audit/c38_density_continuation_20261101'
while not (R/'runs/c38_full_refit/predictions/val_full_c34_fresh_shard1of3/inference.json').exists():time.sleep(3)
time.sleep(5)
with (O/'c38_package.log').open('w') as f:
 p=subprocess.run(['./run_gpu.sh','3','scripts/c38_verify.py'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
(O/'c38_package_exit.json').write_text(json.dumps({'exitcode':p.returncode})+'\n')
