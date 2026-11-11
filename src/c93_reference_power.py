"""Compute additional training-reference examples only; not a fitting or validation job."""
import json,csv,sys,argparse,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts'))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c89_observe import Survival
from src.c91_image import Image
from src.c92_diffuse import Field
from src.c78_power import Components
from c93_geometry import terrain_view,read_compact,read_flow
A=R/'audit/c93_joint_miss'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);plan=np.load(A/'ref_plan.npz')
    ids=plan['ids'][a.shard::3];px=plan['pixels'][a.shard::3];out=A/f'ref_{a.shard}';out.mkdir(exist_ok=True)
    old=Components();surv=Survival();im=Image();diff=Field()
    cs=read_compact();fl=read_flow('screen');rows=list(csv.DictReader(open(DATA/'train_index.csv')))
    h=np.load(DATA/'train_hm.npy',mmap_mode='r');bi={i:j for j,i in enumerate(json.load(open(R/'audit/c88_mesh/base_index.json'))['ids'])}
    base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r');labels=[np.load(DATA/f'train_pl{k}.npy',mmap_mode='r') for k in ('415','58')]
    valid=[np.load(DATA/f'train_mask{k}.npy',mmap_mode='r') for k in ('415','58')]
    fields=[];targets=[];fs=[];ts=time.monotonic()
    for j,i in enumerate(ids):
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);height=h[int(row['hm_row'])];pix=px[j].astype('int64')
        branch=surv.features(height,tr,tc,pix).half().cpu().numpy().astype('float64')
        reflect=im.features(height,tr,tc,pix).half().cpu().numpy().astype('float64')
        scatter,_=diff.features(height,tr,tc,pix,int(i)+920014);first=old.compute(height,tr,tc,pix,int(i)+71019)[:,14:16].astype('float64')
        dd=np.expm1(branch[:,4]).clip(.25)
        rel=10**(-branch[:,[6,11]]/10)+10**(-reflect[:,5:7]/10)+(first+scatter[:,2:4].astype('float64')+scatter[:,4:6])*dd[:,None]**2
        physic=(20*np.log10(dd[:,None]*np.array([415e6,5.775e9])*4*np.pi/299792458)-10*np.log10(rel.clip(1e-15))).astype('float32')
        y=np.stack([v[i].ravel()[pix] for v in labels],1).astype('float32')
        w=np.stack([np.unpackbits(v[i])[pix] for v in valid],1);p=base[bi[i]].reshape(2,-1)[:,pix].T
        tar=np.stack([y,w,p,physic],-1);targets.append(tar)
        ca,kc=cs[i];fa,kf=fl[i];v,f=terrain_view(height,tr,tc,pix,ca[kc],fa[kf]);fields.append(np.stack(list(v.values()),-1).astype('float32'));fs.append(f)
        if j%9==0:print(j+1,len(ids),'seconds',round(time.monotonic()-ts,1),flush=True)
    np.savez_compressed(out/'examples.npz',ids=ids,pixels=px,names=list(v),x=np.stack(fields),fs=np.stack(fs),targets=np.stack(targets))
    atomic_json(out/'done.json',dict(ids=ids.tolist(),gpu=gpu,elapsed=time.monotonic()-ts))
if __name__=='__main__':main()
