"""Reproduce the actual C76 visible-fraction channel on published VAL, independently of the diagnostic ray test."""
import argparse,csv,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA
from src.c70_geometry import GeometricMaps
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--shard',type=int);a=p.parse_args();gpu_guard();A=R/'audit/c78_clear'
data=np.load(A/'val_panel.npz');ids=data['ids'];pix=data['pixels'];pos=np.arange(len(ids))[a.shard::3]
rows=list(csv.DictReader((DATA/'val_index.csv').open()));hh=np.load(DATA/'val_hm.npy',mmap_mode='r');net=GeometricMaps()
v=[]
for j in pos:
    row=rows[ids[j]];h=torch.from_numpy(np.array(hh[int(row['hm_row'])],copy=True)).cuda()
    native,_=net.native(h,int(row['tx_row']),int(row['tx_col']));px=torch.from_numpy(pix[j].astype('int64')).cuda()
    vf=native[:,18].half().float()
    m=(vf[net.ids[px]]*net.w[px]).sum(1).half()
    v.append(m.cpu().numpy())
np.savez(A/f'val_canonical_{a.shard}.npz',positions=pos,values=np.stack(v))
