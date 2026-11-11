import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
gpu=gpu_guard()
sys.path.insert(0,str(R/'.rt_packages'))
import mitsuba as mi
print('variants',mi.variants(),flush=True)
mi.set_variant('cuda_ad_mono_polarized')
import drjit as dr
x=dr.arange(mi.Float,1024);v=dr.sum(x*x);dr.eval(v)
import sionna.rt as rt
print(json.dumps(dict(GPU=gpu,variant=mi.variant(),sionna=rt.__version__,sum=float(v[0]))),flush=True)
atomic_json(R/'audit/c131_transport/rt_backend.json',dict(GPU=gpu,variant=mi.variant(),sionna=rt.__version__,sum=float(v[0])))
