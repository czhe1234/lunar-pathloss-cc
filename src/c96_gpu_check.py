"""Read-only initialization and tiny compute check on one explicitly approved UUID."""
import sys,ctypes,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import GPU_UUID,gpu_guard
d=ctypes.CDLL('libcuda.so.1')
ret=d.cuInit(0);name=ctypes.c_char_p();desc=ctypes.c_char_p()
d.cuGetErrorName(ret,ctypes.byref(name));d.cuGetErrorString(ret,ctypes.byref(desc))
out=dict(requested_uuid=GPU_UUID,cuInit=int(ret),error_name=name.value.decode() if name.value else None,error=desc.value.decode() if desc.value else None)
if ret==0:
 import torch
 out['gpu']=gpu_guard();x=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16);torch.cuda.synchronize()
 t=time.monotonic();y=x@x;torch.cuda.synchronize();out['matmul_seconds']=time.monotonic()-t;out['finite']=bool(torch.isfinite(y).all())
p=R/'audit/c96_information';p.mkdir(parents=True,exist_ok=True);(p/(GPU_UUID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
