"""Common neighbouring RX queries for all retained TX of each FIT terrain."""
import sys,json,csv,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c132_geometry import Geometry,arrays
from src.c133_geometry import Positions
import numpy as np,torch
A=R/'audit/c134_paired'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--of',type=int,default=2);args=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3);plan=json.loads((A/'protocol.json').read_text());rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 ys=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];vs=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
 idx=json.loads((R/'audit/c101_residual/base0of1/indices.json').read_text());bi={i:j for j,i in enumerate(idx)};base=np.load(R/'audit/c101_residual/base0of1/prediction.npy',mmap_mode='r')
 groups={}
 for i in plan['fit']:groups.setdefault(rows[i]['terrain_id'],[]).append(i)
 e=Geometry();pos=Positions(e);start=time.monotonic();count=0
 for t in sorted(groups)[args.shard::args.of]:
  rng=np.random.default_rng(134041+int(t));p=rng.choice(65536,768,replace=False);vertical=rng.integers(0,2,768).astype(bool)
  r=p//256;c=p%256;n=np.where(vertical,((r+1)%256)*256+c,r*256+(c+1)%256);ok=np.where(vertical,r<255,c<255);pixels=np.r_[p,n].astype('int32')
  h=hm[int(rows[groups[t][0]]['hm_row'])];scene=e.scene(h);hit,rx=pos(scene,pixels)
  out=A/'cache'/t;out.mkdir(parents=True,exist_ok=True)
  for i in groups[t]:
   row=rows[i];query=e.query(h,int(row['tx_row']),int(row['tx_col']),pixels,scene)
   # Geometry computation has completed; only now attach separate supervision fields.
   target=np.stack([a[i].reshape(-1)[pixels] for a in ys],-1).astype('float32')
   valid=np.stack([np.unpackbits(a[i])[pixels] for a in vs],-1).astype(bool)
   b=np.array(base[bi[i]]).reshape(2,-1)[:,pixels].T
   np.savez(out/f'{i}.npz',**arrays(query),hitxy=hit.cpu().numpy(),rxxy=rx.cpu().numpy(),pixels=pixels,pair_ok=ok,target=target,valid=valid,base=b)
   count+=1
 atomic_json(A/f'build_{args.shard}.json',dict(status='complete',GPU=gpu,maps=count,seconds=time.monotonic()-start,geometry_inputs='H/TX only; targets attached separately after geometry forward'))
if __name__=='__main__':main()
