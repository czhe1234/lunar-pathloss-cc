from pathlib import Path
import time,subprocess
p=Path('/proc/3767361/stat')
while p.exists() and p.read_text().split()[2]!='Z':time.sleep(2)
for k in range(30):
 code=subprocess.call(['./run_gpu.sh','4','src/c64_height_phase.py'])
 if code!=75:raise SystemExit(code)
 time.sleep(1)
raise SystemExit(75)
