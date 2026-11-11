"""Native hit coordinates supply terrain context to the existing path calculations."""
import sys,json,csv,time,ctypes,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c132_geometry import Geometry
import numpy as np,torch
A=R/'audit/c133_context'
class Positions:
 def __init__(self,engine):
  self.engine=engine;self.lib=ctypes.CDLL(str(R/'src/libc133_positions.so'));P=ctypes.c_void_p;I=ctypes.c_int;F=ctypes.c_float
  self.lib.c133_positions.argtypes=[P,P,P,P,I,F,P,P,P];self.lib.c133_positions.restype=I
 @torch.no_grad()
 def __call__(self,scene,pixels):
  e=self.engine;px=torch.tensor(np.array(pixels),device='cuda',dtype=torch.long);ids=e.rt.ids[px].int().contiguous();w=e.rt.weights[px].contiguous()
  hitxy=torch.empty(len(px),48,2,device='cuda');rxxy=torch.empty(len(px),2,device='cuda')
  err=self.lib.c133_positions(scene['g'].data_ptr(),e.rdirs.data_ptr(),ids.data_ptr(),w.data_ptr(),len(px),scene['top'],hitxy.data_ptr(),rxxy.data_ptr(),torch.cuda.current_stream().cuda_stream)
  if err:raise RuntimeError(err)
  return hitxy,rxxy
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=2);a=p.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3);e=Geometry();pos=Positions(e);rows=list(csv.DictReader((DATA/'train_index.csv').open()));hm=np.load(DATA/'train_hm.npy',mmap_mode='r')
 plan=json.loads((R/'audit/c132_transport/protocol.json').read_text());groups={}
 for i in plan['fit']+plan['monitor']:groups.setdefault(rows[i]['terrain_id'],[]).append(i)
 todo=sorted(groups)[a.shard::a.of];started=time.monotonic();count=0
 for terrain in todo:
  h=hm[int(rows[groups[terrain][0]]['hm_row'])];scene=e.scene(h);out=A/'positions'/terrain;out.mkdir(parents=True,exist_ok=True)
  for i in groups[terrain]:
   old=R/'audit/c132_transport/cache'/terrain/f'{i}.npz';px=np.load(old)['pixels'];hit,rx=pos(scene,px)
   assert torch.isfinite(hit).all() and torch.isfinite(rx).all()
   np.savez(out/f'{i}.npz',hitxy=hit.cpu().numpy(),rxxy=rx.cpu().numpy());count+=1
 atomic_json(A/f'positions_{a.shard}.json',dict(status='complete',GPU=gpu,maps=count,terrains=len(todo),seconds=time.monotonic()-started,labels_read=False))
if __name__=='__main__':main()
