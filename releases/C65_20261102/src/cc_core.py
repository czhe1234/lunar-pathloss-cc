"""Independent lunar radio-map baseline. No imports from the parent project."""
import os
# User authorization 2026-11-01: physical cards 1, 3, 4 only.
# Legacy invocations keep GPU 3 unless the new launcher explicitly selects a UUID.
APPROVED_GPU_UUIDS = frozenset({
    'GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009',
    'GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0',
    'GPU-70b6fceb-c7c4-9467-3c81-097c9284c160',
})
GPU_UUID = os.environ.get('CC_GPU_UUID', 'GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0')
if GPU_UUID not in APPROVED_GPU_UUIDS or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
    raise RuntimeError('Use CC/run_gpu.sh {1|3|4} or legacy CC/run_gpu3.sh; one approved full UUID is required before importing torch.')
from pathlib import Path
import csv, ctypes, hashlib, json, time, uuid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'data'
OFFSET=80.0
SCALE=40.0

def gpu_guard():
    assert torch.cuda.device_count()==1, 'Exactly the approved GPU must be visible'
    driver=ctypes.CDLL('libcuda.so.1')
    assert driver.cuInit(0)==0
    dev=ctypes.c_int();assert driver.cuDeviceGet(ctypes.byref(dev),0)==0
    raw=(ctypes.c_ubyte*16)();assert driver.cuDeviceGetUuid(ctypes.byref(raw),dev)==0
    actual='GPU-'+str(uuid.UUID(bytes=bytes(raw)))
    assert actual==GPU_UUID,(actual,GPU_UUID)
    props=torch.cuda.get_device_properties(0)
    return {'uuid':actual,'logical_device':0,'name':props.name,'memory_bytes':props.total_memory,'torch':torch.__version__,'cuda_runtime':torch.version.cuda}

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

