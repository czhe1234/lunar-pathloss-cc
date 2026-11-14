"""Single C146 controller; shared authorized devices only; paired same-device training."""
from pathlib import Path
import fcntl,json,os,subprocess,time,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT/"audit/c146_solver_ablation";RUN=ROOT/"runs/c146_solver_ablation"
DEVICES={1:"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",3:"GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",4:"GPU-70b6fceb-c7c4-9467-3c81-097c9284c160"}
def status(stage,**kw):
    r={"stage":stage,"controller_pid":os.getpid(),"updated":datetime.now(timezone.utc).isoformat(),**kw}
    p=AUDIT/"controller_status.json";t=p.with_suffix(".tmp");t.write_text(json.dumps(r,indent=2)+"\n");t.replace(p);print(json.dumps(r),flush=True)
def resources():
    output=[]
    for card in (1,4,3):
        uuid=DEVICES[card]
        try:
            v=subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=memory.free,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=10).strip().split(",")
            jobs=subprocess.check_output(["nvidia-smi","-i",uuid,"--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=10).strip()
            output.append({"card":card,"uuid":uuid,"free_MiB":int(v[0]),"used_MiB":int(v[1]),"utilization":int(v[2]),"existing_jobs":jobs.splitlines()})
        except (subprocess.SubprocessError,ValueError,IndexError):continue
    return output
def main():
    lock=(AUDIT/"controller.lock").open("w")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("C146 already controlled")
    if (AUDIT/"comparison_summary.json").exists():status("complete");return
    attempts=0
    while not (RUN/"export_complete.json").exists():
        snapshot=resources();candidates=[r for r in snapshot if r["free_MiB"]>=48000]
        if not candidates:
            status("waiting_for_memory",resources=snapshot,sharing_allowed=True);time.sleep(30);continue
        card=candidates[0]["card"];log=AUDIT/f"training_attempt{attempts+1}.log"
        with log.open("a") as stream:
            p=subprocess.Popen(["./run_gpu.sh",str(card),"src/c146_train.py"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            status("running",card=card,uuid=DEVICES[card],child_pid=p.pid,log=str(log),sharing_allowed=True,resource_at_launch=candidates[0])
            while p.poll() is None:time.sleep(30)
        if p.returncode==75:
            status("waiting_for_CC_device_lock",card=card);time.sleep(30);continue
        attempts+=1
        if p.returncode!=0:
            status("failed",exit_code=p.returncode,log=str(log),recovery="Inspect first; resume jointly from committed epoch, do not start duplicate experiment.")
            return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1")
    with (AUDIT/"verification.log").open("a") as stream:
        p=subprocess.Popen([".venv/bin/python","src/c146_verify.py"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        status("CPU_verification",child_pid=p.pid)
        while p.poll() is None:time.sleep(30)
    status("complete" if p.returncode==0 else "verification_failed",exit_code=p.returncode)
if __name__=="__main__":main()
