"""Merge fixed map-order input-only feature shards; CPU only."""
from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parents[1];A=R/'audit/c124_spectrum';S=R/'audit/c101_residual'
records={}
for split,nshards in [('fit',2),('monitor',1)]:
 d=json.load(open(S/split/'done.json'));maps=d['ids'];ppm=d['points_per_map'];out=A/split
 f=np.lib.format.open_memmap(out/'spectrum.npy','w+',dtype=np.float16,shape=(len(maps)*ppm,256))
 t=np.lib.format.open_memmap(out/'total.npy','w+',dtype=np.float32,shape=(len(maps)*ppm,2));seen=np.zeros(len(maps),bool)
 for k in range(nshards):
  p=out/f'part{k}of{nshards}';done=json.load(open(p/'done.json'));assert done['status']=='complete'
  ids=json.load(open(p/'indices.json'));x=np.load(p/'spectrum.npy',mmap_mode='r');u=np.load(p/'total.npy',mmap_mode='r')
  assert len(x)==len(ids)*ppm
  for j,i in enumerate(ids):
   assert not seen[i];seen[i]=True
   f[i*ppm:(i+1)*ppm]=x[j*ppm:(j+1)*ppm];t[i*ppm:(i+1)*ppm]=u[j*ppm:(j+1)*ppm]
 assert seen.all();f.flush();t.flush()
 records[split]=dict(maps=len(maps),points=len(maps)*ppm,source_ids_sha256=hashlib.sha256(json.dumps(maps).encode()).hexdigest(),sample_min=float(f[::1000].min()),sample_max=float(f[::1000].max()))
 (out/'merged.json').write_text(json.dumps(records[split],indent=2))
 print(split,records[split],flush=True)
(A/'features.json').write_text(json.dumps(records,indent=2))
