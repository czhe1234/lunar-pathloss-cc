import csv,time
from pathlib import Path
import torch,numpy as np
from src.c132_train import Dataset as Parent,tensor
R=Path(__file__).resolve().parents[1];D=R.parent/'data'
class Dataset(Parent):
 def __init__(self,part,ids=None):
  super().__init__(part,ids)
  hm=np.load(D/'train_hm.npy',mmap_mode='r');byterrain={r['terrain_id']:int(r['hm_row']) for r in self.rows}
  self.heights=tensor(hm[[byterrain[t] for t in self.tids]]).float()
  self.txxy=torch.tensor([[2*int(self.rows[i]['tx_col'])/256-1,2*(int(self.rows[i]['tx_row'])+1)/256-1] for i in self.ids],device='cuda')
  for j,i in enumerate(self.ids):
   ar=np.load(R/'audit/c133_context/positions'/self.terrain[j]/f'{i}.npz')
   for k in ['hitxy','rxxy']:
    x=ar[k]
    if k not in self.q:self.q[k]=torch.empty((len(self.ids),*x.shape),device='cuda')
    self.q[k][j]=tensor(x)
 def batch(self,ix,pixels=None):
  g,q,b,y,v=super().batch(ix,pixels)
  ix=torch.as_tensor(ix,device='cuda',dtype=torch.long)
  unique,inverse=torch.unique(self.gid[ix],return_inverse=True)
  q['h']=self.heights[unique];q['inverse']=inverse;q['txxy']=self.txxy[ix]
  return g,q,b,y,v
