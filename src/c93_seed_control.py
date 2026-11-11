"""Hold H/TX constant, alternate physical integration seeds while keeping the C82 and C92 direct/specular terms fixed."""
import sys,csv,json,time,argparse
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'scripts'))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c92_diffuse import Field
from src.c78_power import Components
A=R/'audit/c93_joint_miss'
def main():
    ps=argparse.ArgumentParser();ps.add_argument('--shard',type=int,required=True);ps.add_argument('--kind',choices=['confirm','tx'],required=True);a=ps.parse_args()
    gpu=gpu_guard();torch.set_num_threads(2);h=np.load(DATA/'train_hm.npy',mmap_mode='r');rows=list(csv.DictReader(open(DATA/'train_index.csv')))
    # Physical-only saved columns from C89 and C91 at exactly these coordinates, independent of stochastic diffuse quadrature.
    if a.kind=='confirm':
        v=np.load(A/'confirm_geo.npz');ids=v['ids'][a.shard::3];pixels=v['pixels'][a.shard::3]
        loc={};img={};br={}
        for f in sorted((R/'audit/c91_image').glob('confirm[0-9]')):
            ar=np.load(f/'x.npy',mmap_mode='r');l=np.load(f/'loc.npy');meta=json.load(open(f/'done.json'))['rows']
            for j,row in enumerate(meta):loc[row['index']]=(row['pos'],l[j]);img[row['index']]=ar[j,:,5:7].astype('float64')
        for f in sorted((R/'audit/c89_survival').glob('shard*')):
            ar=np.load(f/'features.npy',mmap_mode='r');meta=json.load(open(f/'done.json'))['positions'];br.update({n:(ar,j) for j,n in enumerate(meta)})
    else:
        from src.c89_observe import Survival
        from src.c91_image import Image
        surv=Survival();image=Image();q=json.load(open(A/'tx_plan.json'));ids=q['ids'][a.shard::3]
        pixels=[np.array(q['pixels'][rows[i]['terrain_id']]) for i in ids]
    old=Components();df=Field();waves=[];ts=time.monotonic()
    for j,(i,px) in enumerate(zip(ids,pixels)):
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);hh=h[int(row['hm_row'])]
        if a.kind=='confirm':
            pos,l=loc[i];a0,k=br[pos];ch=a0[k,l].astype('float64');im=img[i]
        else:
            ch=surv.features(hh,tr,tc,px.astype(np.int64)).half().cpu().numpy().astype('float64')
            im=image.features(hh,tr,tc,px.astype(np.int64)).half().cpu().numpy().astype('float64')[:,5:7]
        d=np.expm1(ch[:,4]).clip(.25);fs=20*np.log10(d[:,None]*np.array([415e6,5.775e9])[None]*4*np.pi/299792458)
        coeff=10**(-ch[:,[6,11]]/10)+10**(-im/10)
        zz=[]
        for seed in ((931901,931907) if a.kind=='confirm' else (931901,)):
            first=old.compute(hh,tr,tc,px.astype(np.int64),seed)[:,14:16].astype('float64')
            multi,_=df.features(hh,tr,tc,px.astype(np.int64),seed+1743)
            rel=coeff+(first+multi[:,2:4].astype('float64')+multi[:,4:6])*d[:,None]**2
            zz.append((fs-10*np.log10(rel.clip(1e-15))).astype('float32'))
        waves.append(zz)
        if j%25==0:print(j+1,len(ids),'seconds',round(time.monotonic()-ts,1),flush=True)
    out=A/(a.kind+'_seed'+str(a.shard));out.mkdir(exist_ok=True);np.save(out/'power.npy',np.array(waves))
    atomic_json(out/'done.json',dict(ids=[int(i) for i in ids],gpu=gpu,seconds=time.monotonic()-ts))
if __name__=='__main__':main()
