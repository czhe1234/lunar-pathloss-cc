from pathlib import Path
import numpy as np,torch
from src.c133_data import Dataset as Parent
from src.c132_train import tensor
R=Path(__file__).resolve().parents[1]
class Dataset(Parent):
 def __init__(self,part,ids=None):
  super().__init__(part,ids)
  self.pair_ok=torch.zeros(len(self.ids),self.pp//2,device='cuda',dtype=torch.bool)
  if part=='fit':
   for j,i in enumerate(self.ids):
    a=np.load(R/'audit/c134_paired/cache'/self.terrain[j]/f'{i}.npz')
    for k in self.q:self.q[k][j]=tensor(a[k])
    self.y[j]=tensor(a['target']);self.v[j]=tensor(a['valid']);self.base[j]=tensor(a['base']);self.pair_ok[j]=tensor(a['pair_ok'])
   self.den=(self.v*self.weights[:,None,None]).sum((0,1))/len(self.ids)
