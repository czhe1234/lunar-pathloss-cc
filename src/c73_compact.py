"""Compact input-only registered terrain and transport descriptors for a teacher-free dense model."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.cc_core import gpu_guard,atomic_json,DATA
from src.c72_native_features import NativeFeatures
import numpy as np
import torch

KEEP=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,39,40,41,46]
@torch.no_grad()
def compact(native,ids,weights):
    # native contains both triangle orientations as consecutive 47-channel fields.
    raw=native[:,:255,:255].reshape(2,47,-1).permute(1,0,2).reshape(47,-1).float()
    read=lambda t: (t[:,ids]*weights[None]).sum(-1)
    mean=read(raw[KEEP])
    linear=read(torch.pow(10.,4*raw[[27,28,33,34]]))
    extra=torch.log10(linear.clamp_min(1e-12))/4
    std=(read(raw[[9,18]].square())-mean[[9,16]].square()).clamp_min(0).sqrt()
    return torch.cat((mean,extra,std),0).reshape(45,256,256)
def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--out',required=True);p.add_argument('--cached',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--of',type=int,default=1)
    a=p.parse_args();gpu=gpu_guard()
    ix=json.load(open(ROOT/a.panel))[a.shard::a.of];out=ROOT/a.out/f'part{a.shard}of{a.of}';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'indices.json',ix);rows=list(csv.DictReader((DATA/'train_index.csv').open()))
    hh=np.load(DATA/'train_hm.npy',mmap_mode='r');engine=NativeFeatures()
    old={}
    if a.cached:
        for d in [d for folder in ['features/c72_native','features/c72_dev_extra_native'] for d in (ROOT/folder).glob('part*')]:
            if (d/'complete.json').is_file():
                src=np.load(d/'features.npy',mmap_mode='r');old.update({i:(src,j) for j,i in enumerate(json.load(open(d/'indices.json')))})
    arr=np.lib.format.open_memmap(out/'features.npy','w+',dtype=np.float16,shape=(len(ix),45,256,256))
    t=time.monotonic()
    for k,i in enumerate(ix):
        r=rows[i]
        if i in old:
            src,j=old[i];v=torch.as_tensor(np.array(src[j],copy=True),device='cuda')
        else:
            v=engine(hh[int(r['hm_row'])],int(r['tx_row']),int(r['tx_col'])).half()
        val=compact(v,engine.geo.ids,engine.geo.w).half().cpu().numpy()
        if not np.isfinite(val).all():raise RuntimeError(i)
        arr[k]=val
        if k%120==0 or k+1==len(ix):
            h={'maps':k+1,'total':len(ix),'seconds':round(time.monotonic()-t,1)};atomic_json(out/'progress.json',h);print(h,flush=True)
    arr.flush();atomic_json(out/'complete.json',dict(vars(a),gpu=gpu,num_channels=45))
if __name__=='__main__':main()
