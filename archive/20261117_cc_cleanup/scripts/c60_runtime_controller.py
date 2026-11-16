import time,subprocess
from pathlib import Path
pid=3679655
while Path('/proc/'+str(pid)+'/stat').exists() and Path('/proc/'+str(pid)+'/stat').read_text().split()[2]!='Z':time.sleep(1)
for trial in range(30):
 code=subprocess.call(['./run_gpu.sh','4','src/c60_runtime_compile.py'])
 if code!=75:raise SystemExit(code)
 time.sleep(1)
raise SystemExit(75)
