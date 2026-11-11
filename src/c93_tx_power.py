"""Query a single physical field per fixed TRAIN terrain/TX, at identical RX positions for nearby TXs."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,DATA,atomic_json
from src.c89_observe import Survival
from src.c91_image import Image
from src.c92_diffuse import Field
from src.c78_power import Components
A=R/'audit/c93_joint_miss'
def main():
    ps=argparse.ArgumentParser();ps.add_argument('--shard',required=True,type=int);a=ps.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);z=json.load(open(A/'tx_plan.json'));ids=z['ids'][a.shard::3]
    rows=list(csv.DictReader(open(DATA/'train_index.csv')));H=np.load(DATA/'train_hm.npy',mmap_mode='r');ph=[]
    old=Components();sf=Survival();im=Image();df=Field();ts=time.monotonic()
    for j,i in enumerate(ids):
        row=rows[i];pix=np.array(z['pixels'][row['terrain_id']]);tr,tc=int(row['tx_row']),int(row['tx_col']);h=H[int(row['hm_row'])]
        branch=sf.features(h,tr,tc,pix).half().cpu().numpy().astype('float64');mirror=im.features(h,tr,tc,pix).half().cpu().numpy().astype('float64')
        mult,_=df.features(h,tr,tc,pix,i+920014);scat=old.compute(h,tr,tc,pix,i+71019)[:,14:16].astype('float64')
        d=np.expm1(branch[:,4]).clip(.25);pw=10**(-branch[:,[6,11]]/10)+10**(-mirror[:,5:7]/10)+(scat+mult[:,2:4].astype('float64')+mult[:,4:6])*d[:,None]**2
        ph.append(20*np.log10(d[:,None]*np.array([415e6,5.775e9])[None]*4*np.pi/299792458)-10*np.log10(pw.clip(1e-15)))
        if j%13==0:print(j+1,len(ids),round(time.monotonic()-ts,1),flush=True)
    dest=A/('tx'+str(a.shard));dest.mkdir(exist_ok=True);np.save(dest/'power.npy',np.array(ph,dtype='float32'))
    atomic_json(dest/'done.json',dict(ids=ids,gpu=gpu,elapsed=time.monotonic()-ts))
if __name__=='__main__':main()
