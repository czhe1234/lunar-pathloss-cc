import os,time,json,signal
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
R=Path(__file__).resolve().parents[1];run=R/'runs/c21_pair_s20261123';audit=R/'audit/c21_execution_20261101'
while True:
 try:
  e=json.loads((run/'latest_eval.json').read_text())
  if e['step']>=6000:
   steps=[]
   for label in ['A','B']:
    ck=torch.load(run/f'last_{label}.pt',map_location='cpu',weights_only=False);steps.append(ck['step']);del ck
   if min(steps)>=6000:break
 except (OSError,EOFError,RuntimeError,ValueError):pass
 time.sleep(2)
candidates=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().split(b'\0')
  if len(cmd)>1 and cmd[1]==b'src/c21_pair.py' and Path(os.readlink(p/'cwd'))==R:
   st=(p/'status').read_text();ppid=int(next(l.split(':')[1] for l in st.splitlines() if l.startswith('PPid:')));candidates.append((int(p.name),ppid))
 except (OSError,StopIteration):pass
pidset={i for i,p in candidates};masters=[i for i,p in candidates if p not in pidset];assert len(masters)==1,candidates
os.kill(masters[0],signal.SIGINT)
record={'status':'stopped_after_matching_checkpoints_for_architecture_diagnosis','checkpoint_steps':steps,'pid':masters[0],'reason':'At3000, changing terrain tokens shifts predictions only~0.01dB and removing geometry bias <0.0001dB on fourDEV maps. Preserve6000 checkpoints and investigate inactive global branch before spending remaining18000 budget. This is an unplanned diagnostic stop, not a completed18000-step negative comparison.','unix':time.time()}
(audit/'matched_stop.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
