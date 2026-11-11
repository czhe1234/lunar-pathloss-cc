"""Non-disruptive CUDA recovery check, restricted to one authorized GPU UUID."""
import sys,os,json,time,ctypes
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import GPU_UUID,gpu_guard
out=dict(requested_uuid=GPU_UUID,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),uptime_seconds=float(Path('/proc/uptime').read_text().split()[0]))
d=ctypes.CDLL('libcuda.so.1');ret=d.cuInit(0);out['cuInit']=int(ret)
try:
 fd=os.open('/dev/nvidia-uvm',os.O_RDWR);os.close(fd);out['uvm_open']='OK'
except OSError as e:out['uvm_open']=dict(errno=e.errno,error=str(e))
if ret==0:
 import torch
 out['gpu']=gpu_guard();torch.manual_seed(980411)
 a=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 b=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 t=time.monotonic();z=a@b;loss=z.float().square().mean();loss.backward();torch.cuda.synchronize()
 out.update(forward_backward_finite=bool(torch.isfinite(z).all() and torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()),
 elapsed_seconds=time.monotonic()-t,peak_memory_bytes=torch.cuda.max_memory_allocated())
 del a,b,z,loss;torch.cuda.empty_cache()
p=R/'audit/c98_cuda_recovery';p.mkdir(parents=True,exist_ok=True)
(p/(GPU_UUID+'.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
if ret or not out.get('forward_backward_finite',False):raise SystemExit(1)
