"""C32 cached teacher base with public H/TX features; labels are separate."""
from pathlib import Path
import json
import numpy as np
import torch
from c21_data import Data
R=Path(__file__).resolve().parents[1]
class Maps(torch.utils.data.Dataset):
 def __init__(self,ids,part):
  self.ids=np.array(ids);self.data=Data();self.part=part
  if part=="fit":
   meta=json.loads((R/"features/c17_spatial/metadata.json").read_text());self.lookup={i:j for j,i in enumerate(meta["ids"]["fit"])}
   self.base=np.load(R/"features/c17_spatial/fit_base.npy",mmap_mode="r")
  else:
   dest=R/"runs/c10_points_phase_geometry_s20261101/predictions/train_dev_gpu_c10_fast_fresh"
   self.lookup={i:j for j,i in enumerate(json.loads((dest/"indices.json").read_text())["index_rows"])}
   self.base=[np.load(dest/("pl"+b+".npy"),mmap_mode="r") for b in ["415","58"]]
 def __len__(self):return len(self.ids)
 def __getitem__(self,k):
  i=int(self.ids[k]);x,tx,fs,y,m=self.data.sample(i);j=self.lookup[i]
  base=np.array(self.base[j],np.float32,copy=True) if self.part=="fit" else np.stack([v[j] for v in self.base]).astype(np.float32)
  x=np.concatenate((x,(base-fs)/20),0)
  return i,x,tx,base,y,m
def loader(ids,part,batch=8,workers=4):
 return torch.utils.data.DataLoader(Maps(ids,part),batch_size=batch,num_workers=workers,pin_memory=True,shuffle=False)
