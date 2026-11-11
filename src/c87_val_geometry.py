"""Recompute the frozen C76/C82 visibility grouping for all official validation pixels."""
from pathlib import Path
import csv, json, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import DATA, atomic_json, gpu_guard
from src.c70_geometry import GeometricMaps

def main():
    gpu=gpu_guard();torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.06)
    out=ROOT/'audit/c87_val';out.mkdir(exist_ok=True)
    c78=ROOT/'audit/c78_clear';panel=np.load(c78/'val_panel.npz')
    points=panel['pixels']; rows=list(csv.DictReader((DATA/'val_index.csv').open()))
    assert len(rows)==2330 and np.array_equal(panel['ids'],np.arange(len(rows)))
    prior=np.empty((len(rows),2048),np.float16);seen=np.zeros(len(rows),bool)
    for p in sorted(c78.glob('val_canonical_*.npz')):
        z=np.load(p);ix=z['positions'];assert not seen[ix].any();prior[ix]=z['values'];seen[ix]=True
    assert seen.all()
    heights=np.load(DATA/'val_hm.npy',mmap_mode='r');eng=GeometricMaps()
    tmp=out/'visibility_class.partial.npy'
    m=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(rows),256,256))
    mismatch=0;start=time.monotonic()
    for i,row in enumerate(rows):
        h=torch.from_numpy(np.array(heights[int(row['hm_row'])],copy=True)).cuda()
        v,_=eng.native(h,int(row['tx_row']),int(row['tx_col']))
        fr=(v[:,18].half().float()[eng.ids]*eng.w).sum(1).half().cpu().numpy()
        mismatch+=int(np.count_nonzero(fr[points[i]]!=prior[i]))
        val=fr.astype(np.float32)
        m[i]=np.where(val>.99,0,np.where(val>=.01,1,2)).reshape(256,256)
        if (i+1)%500==0:print(i+1,'of',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    assert mismatch==0, mismatch
    m.flush();del m;tmp.replace(out/'visibility_class.npy')
    atomic_json(out/'geometry.json',dict(gpu=gpu,maps=len(rows),full_pixel_evaluations=len(rows)*65536,
        identical_sampled_visibility_values=int(points.size),mismatches=mismatch,
        zones=['clear (>0.99)','partial [0.01,0.99]','blocked (<0.01)'],seconds=time.monotonic()-start))
if __name__=='__main__':main()
