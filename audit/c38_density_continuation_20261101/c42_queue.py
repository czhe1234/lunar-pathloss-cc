from pathlib import Path
import subprocess,time,json
R=Path(__file__).resolve().parents[2];O=R/'audit/c38_density_continuation_20261101'
paths=[R/'features/c42_rx_patch/complete.json',R/'runs/c38_full_refit/predictions/val_full_c34_fresh_shard0of3/inference.json']
for p in paths:
 while not p.exists():time.sleep(3)
time.sleep(5)
with (O/'c42_train.log').open('w') as f:
 p=subprocess.run(['./run_gpu.sh','1','src/c42_train.py','--mode','periodic','--members','8','--width','1024','--dense','--steps','20000','--name','c42_rx_patch'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
(O/'c42_train_exit.json').write_text(json.dumps({'exitcode':p.returncode})+'\n')
