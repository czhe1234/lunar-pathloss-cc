"""Use each authorized card for replay once its matching training process finishes."""
from pathlib import Path
import subprocess,time,json,concurrent.futures
R=Path(__file__).resolve().parents[1]
def work(card,v):
    marker=R/'runs'/f'c43_{v}_fit256'/'result.json'
    deadline=time.time()+3600
    while not marker.exists():
        assert time.time()<deadline, f"Timed out waiting for {v}"
        time.sleep(15)
    time.sleep(1)
    for stage in ['fit64','fit256']:
        name=f'c43_{v}_{stage}'
        with (R/'logs'/f'{name}_fp32.log').open('x') as log:
            subprocess.run(['./run_gpu.sh',str(card),'scripts/c43_fp32.py','--run',name],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'replay_complete':v}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(work,c,v) for c,v in [(4,'cnn'),(1,'grid'),(3,'path')]]
    for f in concurrent.futures.as_completed(futures):f.result()
