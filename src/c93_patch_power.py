"""Reconstruct the same frozen physical estimator in adjacent 7x7 cells using H and TX only."""
import json,csv,sys,argparse,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c89_observe import Survival
from src.c91_image import Image
from src.c92_diffuse import Field
from src.c78_power import Components
A=R/'audit/c93_joint_miss'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',required=True,type=int);a=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);plan=json.load(open(A/'patches.json'));ids=plan['terrain_indices'][a.shard::3]
    rows=list(csv.DictReader(open(DATA/'train_index.csv')));h=np.load(DATA/'train_hm.npy',mmap_mode='r')
    old=Components();surv=Survival();image=Image();diff=Field();out=A/('patch_power'+str(a.shard));out.mkdir(exist_ok=True)
    ts=time.monotonic();fields=[]
    for j,i in enumerate(ids):
        r=rows[i];tr,tc=int(r['tx_row']),int(r['tx_col']);height=h[int(r['hm_row'])];pixels=np.array(plan['maps'][str(i)]['px'],np.int64)
        branch=surv.features(height,tr,tc,pixels).half().cpu().numpy().astype('float64')
        im=image.features(height,tr,tc,pixels).half().cpu().numpy().astype('float64')
        multi,_=diff.features(height,tr,tc,pixels,i+920014)
        first=old.compute(height,tr,tc,pixels,i+71019)[:,14:16].astype('float64')
        dist=np.expm1(branch[:,4]).clip(.25)
        rel=10**(-branch[:,[6,11]]/10)+10**(-im[:,5:7]/10)
        rel+=(first+multi[:,2:4].astype('float64')+multi[:,4:6])*dist[:,None]**2
        p=20*np.log10(dist[:,None]*np.array([415e6,5.775e9])*4*np.pi/299792458)-10*np.log10(rel.clip(1e-15))
        assert np.isfinite(p).all();fields.append(p.astype('float32'))
        if j%12==0:print(j+1,len(ids),'seconds',round(time.monotonic()-ts,2),flush=True)
    np.save(out/'p.npy',np.array(fields));atomic_json(out/'done.json',dict(ids=ids,gpu=gpu,elapsed=time.monotonic()-ts))
if __name__=='__main__': main()
