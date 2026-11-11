"""Advance the wider local model only on its prespecified held-out criteria."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    new=R/'runs/c75_wide_observed';old=R/'runs/c72_observed'
    local=json.load(open(new/'result.json'));ref=json.load(open(old/'result.json'))
    diff=[b-v for b,v in zip(ref['confirmation'],local['confirmation'])]
    advance=min(diff)>=.02
    decision=dict(confirmation_new=local['confirmation'],confirmation_old=ref['confirmation'],gain=diff,advance_to_complete_DEV=advance)
    write(new/'promotion.json',decision);print(json.dumps(decision),flush=True)
    if not advance:return
    for _ in range(1000):
        cmd=['./run_gpu.sh','1','src/c72_score_dev.py','--run',new.name,'--features','features/c72_native+features/c72_dev_extra_native']
        rc=subprocess.run(cmd,cwd=R).returncode
        if rc!=75:break
        time.sleep(3)
    else:raise TimeoutError('GPU 1 not available')
    if rc:raise RuntimeError(f'full internal evaluation exited {rc}')
    a=json.load(open(new/'full_internal/result.json'));b=json.load(open(old/'full_internal/result.json'))
    more=[u-v for u,v in zip(b['all612']['new'],a['all612']['new'])]
    decision.update(full_gain_over_C72=more,joint_complete_DEV_improvement=min(more)>0)
    write(new/'promotion.json',decision);print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
