"""Use CC's guarded cards only when their complete C82 prediction shard has finished."""
import json,subprocess,time,sys,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'audit/c84_tx'
def run(card,script,args,tag):
    with (R/'logs'/tag).open('a') as f:
        for _ in range(900):
            rc=subprocess.run(['./run_gpu.sh',str(card),script,*args],cwd=R,stdout=f,stderr=subprocess.STDOUT).returncode
            if rc!=75:break
            time.sleep(2)
    if rc:raise RuntimeError(f'{tag} exited {rc}')
def prep(card,j):
    chk=R/f'runs/c82_selected/predictions/val_full_c34_fresh_shard{j}of3/inference.json'
    for _ in range(1800):
        if chk.exists():break
        time.sleep(2)
    else:raise RuntimeError(f'missing C82 shard {j}')
    if not (R/f'features/c84_extra/part{j}of3/complete.json').exists():
        run(card,'src/c73_compact.py',['--panel','audit/c84_tx/extra.json','--out','features/c84_extra','--shard',str(j),'--of','3'],f'c84_cache{j}.log')
def train(card,arm):
    if not (R/f'runs/c84_{arm}/complete.json').exists():
        run(card,'src/c84_train.py',['--arm',arm],f'c84_{arm}.log')
def main():
    with ThreadPoolExecutor(3) as ex:
        jobs=[ex.submit(prep,c,j) for j,c in enumerate([1,3,4])]
        for future in jobs:future.result()
    with ThreadPoolExecutor(3) as ex:
        jobs=[ex.submit(train,c,m) for c,m in [(1,'old'),(3,'new'),(4,'delta')]]
        for future in jobs:future.result()
    scores={}
    for arm in ('old','new','delta'):
        h=json.load(open(R/f'runs/c84_{arm}/history.json'))
        scores[arm]=min(sum(q[0] for q in item['rmse'])/2 for item in h)
    winner=min(scores,key=scores.get)
    (A/'selected.json').write_text(json.dumps(dict(arm=winner,score=scores,source='48 TRAIN monitor terrains before checking confirmation'),indent=2)+'\n')
    # Run the selected treatment and its original-pairs control on the complete held-out panel.
    controls=list(dict.fromkeys([winner,'old','new']));devs=[3,1,4]
    with ThreadPoolExecutor(len(controls)) as ex:
        calls=[ex.submit(run,devs[j],'src/c84_train.py',['--arm',m,'--confirm'],f'c84_{m}_confirm.log') for j,m in enumerate(controls) if not (R/f'runs/c84_{m}/confirm.json').exists()]
        for future in calls:future.result()
    print('selected',winner,'raw TRAIN confirm',json.load(open(R/f'runs/c84_{winner}/confirm.json')),flush=True)
if __name__=='__main__':main()
