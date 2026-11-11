from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl, json, os, subprocess, sys

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
AUDIT=Path(__file__).parent
EXPECTED={
    "1":"GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
    "3":"GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
    "4":"GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
}
probe = """
import json, os, subprocess, sys
sys.path.insert(0, "src")
from cc_core import gpu_guard
import torch
info=gpu_guard()
torch.cuda.set_per_process_memory_fraction(0.15)
x=torch.ones((128,128),device="cuda",requires_grad=True)
y=(x@x).mean()
y.backward()
torch.cuda.synchronize()
assert y.item()==128.0
assert torch.allclose(x.grad,torch.full_like(x,2.0/128))
info["forward_backward_ok"]=True
info["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
info["bf16_supported"]=torch.cuda.is_bf16_supported()
info["visible_uuid"]=os.environ["CUDA_VISIBLE_DEVICES"]
child=subprocess.check_output([sys.executable,"-c","import os;print(os.environ['CUDA_VISIBLE_DEVICES'])"],text=True).strip()
assert child==info["uuid"]
info["child_uuid_inherited"]=True
print(json.dumps(info))
"""
def check(card):
    proc=subprocess.run(["./run_gpu.sh",card,"-c",probe],capture_output=True,text=True,timeout=90)
    if proc.returncode: raise RuntimeError((card,proc.returncode,proc.stdout,proc.stderr))
    result=json.loads(proc.stdout)
    assert result["uuid"]==EXPECTED[card]
    result["physical_card"]=int(card)
    return result

subprocess.run(["bash","-n","run_gpu.sh"],check=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    devices=list(pool.map(check,EXPECTED))
for d in devices: print(json.dumps(d),flush=True)

legacy=subprocess.run(["./run_gpu3.sh","-c",probe],capture_output=True,text=True,timeout=90,
    env={k:v for k,v in os.environ.items() if k!="CC_GPU_UUID"})
assert legacy.returncode==0,legacy.stderr
legacy_info=json.loads(legacy.stdout)
assert legacy_info["uuid"]==EXPECTED["3"]

negative=[]
for card in ["0","2","5","6","7","1,3",""]:
    r=subprocess.run(["./run_gpu.sh",card,"-c","raise RuntimeError('must not execute')"],capture_output=True,text=True)
    assert r.returncode==2,(card,r.returncode)
    negative.append({"selection":card,"exit_code":r.returncode})

# These imports must fail before torch is imported or any CUDA work starts.
for selected,visible in [
    ("GPU-not-authorized","GPU-not-authorized"),
    (EXPECTED["1"],EXPECTED["3"]),
    (EXPECTED["1"],EXPECTED["1"]+","+EXPECTED["3"]),
]:
    env=dict(os.environ,CC_GPU_UUID=selected,CUDA_VISIBLE_DEVICES=visible)
    code="import sys;sys.path.insert(0,'src');import cc_core"
    r=subprocess.run([".venv/bin/python","-c",code],env=env,capture_output=True,text=True)
    assert r.returncode!=0 and "one approved full UUID" in r.stderr,r.stderr
    negative.append({"selection":selected,"visible":visible,"guard_rejected":True})

with open(".gpu1.lock","a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r=subprocess.run(["./run_gpu.sh","1","-c","raise RuntimeError('must not execute')"],capture_output=True,text=True)
    assert r.returncode==75,r.stderr

result={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"devices":devices,
        "legacy_gpu3":legacy_info,"negative_checks":negative,"same_card_lock_exit_code":75,
        "new_training_started":False,"all_passed":True}
(AUDIT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print("ALL CHECKS PASSED",flush=True)
