"""Run frozen inference, verify coverage and call both scoring implementations."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];a=R/'audit/c72_continuation'
def main():
    required=[a/'final_raw_interface/result.json',a/'full_H100/result.json',a/'full_A800/result.json']
    for _ in range(400):
        if all(x.exists() for x in required):break
        time.sleep(2)
    else:raise TimeoutError('raw interface or measured timings still missing')
    qa,h,amp=[json.load(open(v)) for v in required]
    assert max(max(r['cached_replay_RMS_diff']) for r in qa['records'])<.25
    assert min(h['warm_mean'],amp['warm_mean'])<.5,'Measured end-to-end mean exceeds the bound on both devices'
    procs=[]
    for card,shard in [(1,0),(3,1),(4,2)]:
        log=open(R/f'logs/c76_val{shard}.log','a')
        args=['./run_gpu.sh',str(card),'src/c76_predict.py','--model','c76_selected','--shard',str(shard),'--of','3']
        procs.append([subprocess.Popen(args,cwd=R,stdout=log,stderr=subprocess.STDOUT),args,log])
    for item in procs:
        pp,args,log=item;rc=pp.wait()
        for _ in range(100):
            if rc!=75:break
            time.sleep(2);rc=subprocess.run(args,cwd=R,stdout=log,stderr=subprocess.STDOUT).returncode
        log.close()
        if rc:raise RuntimeError(f'{args} exited {rc}')
    for key,args in [('independent',['scripts/c34_score.py','--run','c76_selected','--split','val','--shards','3']),('official',['scripts/c34_official.py','--run','c76_selected','--shards','3'])]:
        with open(R/f'logs/c76_{key}_score.log','w') as f:subprocess.run(['.venv/bin/python',*args],cwd=R,check=True,stdout=f,stderr=subprocess.STDOUT)
    result=json.load(open(R/'audit/c76_selected_official/verified_summary.json'))
    print(json.dumps(dict(RMSE=result['RMSE'],combined=result['combined_rmse_db_masked'],goal=result['goal_both_le_1db'])),flush=True)
if __name__=='__main__':main()
