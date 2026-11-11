"""H/TX-only native path probe; fixed points come from the existing frozen protocol."""
import sys,time,csv,json,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c135_paths import Paths,A
import numpy as np,torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=3)
 p.add_argument('--source',type=int,default=4);p.add_argument('--receiver',type=int,default=1);p.add_argument('--directions',type=int,default=64)
 p.add_argument('--check-count',type=int,default=0);p.add_argument('--tag',default='cache');a=p.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3);engine=Paths(source=a.source,receiver=a.receiver,directions_count=a.directions)
 plan=json.loads((A/'protocol.json').read_text());rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 lookup={};maps=[]
 for part in ['fit','monitor']:
  root=R/'audit/c101_residual'/part;meta=json.loads((root/'done.json').read_text());pp=meta['points_per_map']
  px=np.load(root/'pixel.npy',mmap_mode='r');partids=plan['fit_probe' if part=='fit' else 'monitor']
  index={i:j for j,i in enumerate(meta['ids'])}
  if a.check_count:partids=partids[:a.check_count]
  for i in partids:lookup[i]=np.asarray(px[index[i]*pp:(index[i]+1)*pp]);maps.append((part,i))
 out=A/a.tag;out.mkdir(parents=True,exist_ok=True);times=[];t=time.monotonic()
 for j,(part,i) in enumerate(maps[a.shard::a.of]):
  row=rows[i];h=hm[int(row['hm_row'])];px=lookup[i];tr=int(row['tx_row']);tc=int(row['tx_col'])
  torch.cuda.synchronize();start=time.monotonic();s=engine.forward(h,tr,tc);q=engine.query(s,px);d=engine.direct(s,h,tr,tc,px)
  val=torch.cat((d[:,:2,None].transpose(1,2),q),1)
  assert torch.isfinite(val).all() and (val>=0).all(),i
  ar=val.cpu().numpy();times.append(time.monotonic()-start)
  np.savez(out/f'{i}.npz',pixels=px,power=ar,direct_geometry=d[:,2:].cpu().numpy())
  if j%40==0:
   status=dict(maps=j+1,total=len(maps[a.shard::a.of]),seconds=time.monotonic()-t,median_map_seconds=float(np.median(times)),last_map=i)
   atomic_json(A/f'{a.tag}_{a.shard}_progress.json',status);print(status,flush=True)
 done=dict(status='complete',GPU=gpu,maps=len(times),seconds=time.monotonic()-t,median_map_seconds=float(np.median(times)),parameters=vars(a),
  channels=['direct']+[f'spec_{j+1}' for j in range(engine.pre)]+['diffuse_direct','spec_before_diffuse','spec_after_diffuse','spec_before_and_after_diffuse'],
  label_access=False,mask_access=False)
 atomic_json(A/f'{a.tag}_{a.shard}_done.json',done);print(done,flush=True)
if __name__=='__main__':main()
