import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") in {
    "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
    "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
    "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
}
import ctypes, json
driver=ctypes.CDLL("libcuda.so.1")
code=driver.cuInit(0)
name=ctypes.c_char_p()
description=ctypes.c_char_p()
driver.cuGetErrorName(code,ctypes.byref(name))
driver.cuGetErrorString(code,ctypes.byref(description))
print(json.dumps({"cuInit":code,"name":name.value.decode(),"description":description.value.decode()}))
