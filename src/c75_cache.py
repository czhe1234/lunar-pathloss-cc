"""Cache frozen base predictions and native features on additional disjoint FIT terrains."""
import csv,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from src.cc_core import DATA,atomic_json,gpu_guard
from src.c74_inference import baseline_fields
from src.c60_inference import Predictor
from src.c74_features import FastDescriptors
from src.c72_native_features import NativeFeatures
import numpy as np
import torch
@torch.inference_mode()
def main():
    g=gpu_guard();ids=json.load(open(R/'audit/c72_continuation/extra_fit_c75.json'))
    out=R/'features/c75_fit_extension/fit0of1';out.mkdir(parents=True,exist_ok=False);atomic_json(out/'indices.json',ids)
    base=Predictor(R/'runs/c54_combo_dev');base.features=FastDescriptors();nf=NativeFeatures()
    pr=np.lib.format.open_memmap(out/'prediction.npy','w+',dtype=np.float32,shape=(len(ids),2,256,256))
    fe=np.lib.format.open_memmap(out/'features.npy','w+',dtype=np.float16,shape=(len(ids),94,256,256))
    rows=list(csv.DictReader((DATA/'train_index.csv').open()));hh=np.load(DATA/'train_hm.npy',mmap_mode='r');start=time.monotonic()
    for j,i in enumerate(ids):
        row=rows[i];h=np.array(hh[int(row['hm_row'])],np.float32);tr=int(row['tx_row']);tc=int(row['tx_col'])
        ht=torch.as_tensor(h,device='cuda')[None];b0,_=baseline_fields(base,h,ht,tr,tc);pr[j]=b0[0].cpu().numpy()
        x=nf(h,tr,tc).half().cpu().numpy();assert np.isfinite(x).all();fe[j]=x
        if j%80==0 or j+1==len(ids):
            v=dict(maps=j+1,total=len(ids),seconds=round(time.monotonic()-start,2));atomic_json(out/'progress.json',v);print(v,flush=True)
    pr.flush();fe.flush();atomic_json(out/'complete.json',dict(complete=True,maps=len(ids),gpu=g,input_only=True))
if __name__=='__main__':main()
