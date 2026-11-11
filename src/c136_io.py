import sys,time,json,numpy as np,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard
print(gpu_guard(),flush=True)
ids=json.loads((R/'audit/c136_native/protocol.json').read_text())['monitor'][100:116]
for i in ids:
 t=time.monotonic()
 with np.load(R/'audit/c136_native/cache'/f'{i}.npz') as a:d={k:np.array(a[k]) for k in ['si','sw','ai','aw','dw','uf','inv','pw','feat','active','txxy','df']}
 t1=time.monotonic();g={k:torch.as_tensor(v,device='cuda') for k,v in d.items()};torch.cuda.synchronize();t2=time.monotonic()
 print(i, 'read',round(t1-t,4),'copy',round(t2-t1,4),'MB',sum(x.nbytes for x in d.values())/1e6,flush=True)
