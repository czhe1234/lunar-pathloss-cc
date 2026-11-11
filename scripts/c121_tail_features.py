"""Expose existing unaveraged C72 features. No new geometry or labels as model inputs."""
import sys,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.c121_pairs import Reader,mm
A=R/'audit/c121_pairs'
def main():
 rd=Reader()
 for part in ['fit','monitor']:
  P=R/'audit/c101_residual'/part;allrows=mm(R/f'audit/c120_sign/{part}_rows.npy');maps=mm(P/'mapid.npy');pixel=mm(P/'pixel.npy')
  keep=np.array([int(i) in rd.native for i in maps[allrows]]);rows=np.array(allrows[keep]);ids=np.array(maps[rows])
  out=A/('tail_'+part);out.mkdir(exist_ok=True);np.save(out/'rows.npy',rows)
  n=len(rows);native=np.lib.format.open_memmap(out/'donors.npy',mode='w+',dtype='float32',shape=(n,150))
  start=time.monotonic()
  for j,i in enumerate(np.unique(ids)):
   idx=np.flatnonzero(ids==i);v,meta=rd.donors(int(i),np.array(pixel[rows[idx]]));native[idx]=np.c_[v.reshape(len(idx),141),meta]
   if j%200==0:print(part,j,'of',len(set(ids)),'seconds',time.monotonic()-start,flush=True)
  native.flush()
  meta=dict(rows=n,maps=len(set(ids)),terrains=len(set(rd.rows[int(i)]['terrain_id'] for i in ids)),
   candidate_maps=len(set(maps[allrows])),original_tail_rows=len(allrows),data='same fixed C101 sampled TRAIN points; available native cache intersection',
   value_inputs='H/TX-only native descriptors; selection uses truth tail for diagnosis only',seconds=time.monotonic()-start)
  (out/'done.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True)
if __name__=='__main__':main()
