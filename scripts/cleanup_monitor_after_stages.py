"""Stop only this task's sampling monitor after the bounded stage controller exits."""
import os,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
controller=int((ROOT/'logs/stage_controller.pid').read_text());monitor=int((ROOT/'logs/gpu3_monitor.pid').read_text())
while True:
 try:os.kill(controller,0)
 except ProcessLookupError:break
 time.sleep(10)
state=json.loads((ROOT/'logs/stage_controller.json').read_text());cmd=Path(f'/proc/{monitor}/cmdline')
result={'controller_final_state':state['state'],'time':time.time(),'monitor_pid':monitor,'stopped':False}
if cmd.exists():
 parts=cmd.read_bytes().split(b'\0')
 if parts[0].endswith(b'nvidia-smi') and b'GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0' in parts and any(p.startswith(b'--query-gpu=timestamp,uuid,utilization.gpu') for p in parts):
  os.kill(monitor,signal.SIGTERM);result['stopped']=True
(ROOT/'logs/monitor_cleanup.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
