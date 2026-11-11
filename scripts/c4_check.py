import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import gpu_guard,ROOT,atomic_json,d4,d4_coords
from c4_geometry import extra_geometry
import torch
info=gpu_guard();h=torch.zeros(1,256,256,device='cuda');r=torch.tensor([128],device='cuda');c=r.clone();g=extra_geometry(h,r,c);assert torch.isfinite(g).all() and (g[:,2:]==0).all()
h[:,:,150:155]=20;g=extra_geometry(h,r,c);assert g[0,2,128,200]>0 and g[0,2,128,140]==0
shift=extra_geometry(h+37,r,c);assert (g-shift).abs().max()<.002
rr,cc=d4_coords(r,c,1);rot=extra_geometry(d4(h,1),rr,cc);diff=float((rot-d4(g,1)).abs().max());assert diff<.002
atomic_json(ROOT/'audit/c4_checks.json',{'passed':True,'flat_ridge_translation_rotation':True,'D4_max_difference':diff,'device':info});print('C4 geometry checks passed',diff)
