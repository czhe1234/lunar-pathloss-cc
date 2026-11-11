from pathlib import Path
import time,json
t=time.time();total=0
for name in ['features/c3/train.npy','features/c4_extra/train.npy']:
 n=0
 with Path(name).open('rb') as f:
  while True:
   b=f.read(64*1024*1024)
   if not b:break
   n+=len(b)
 total+=n
 print(json.dumps({'file':name,'bytes':n,'seconds':time.time()-t}),flush=True)
Path('audit/c63_dynamic_full_20261102/prefetch.json').write_text(json.dumps({'complete':True,'bytes':total,'seconds':time.time()-t,'purpose':'Warm shared OS cache for existing H/TX-only geometry; no GPU, labels or masks read'}))
