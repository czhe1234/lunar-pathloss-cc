"""Reconstruct the frozen C82 development prediction once for paired controls."""
import sys,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c82_blend import Data,field_features,gpu_guard,atomic_json
import torch
@torch.inference_mode()
def main():
    gpu=gpu_guard();torch.set_num_threads(2)
    D=Data();A=R/'audit/c88_mesh';cfg=json.load(open(R/'runs/c82_selected/spectrum.json'))
    beta=[torch.tensor(v,device='cuda') for v in cfg['beta']]
    dest=A/'base.partial.npy'
    pred=np.lib.format.open_memmap(dest,mode='w+',shape=(len(D.ix),2,256,256),dtype='float32')
    info={'ids':D.ix,'groups':D.groups,'gpu':gpu};t=time.monotonic()
    se=np.zeros((2,2));n=0
    for i in range(0,len(D.ix),2):
        j=list(range(i,min(i+2,len(D.ix))))
        base,local,graph,fs,vis,y,w=D.batch(j)
        mean=(local+graph)*.5
        corr=torch.cat([(field_features(base,local,graph,fs,vis,cfg['mode'],b)*beta[b][None,:,None,None]).sum(1,keepdim=True) for b in (0,1)],1).clamp(-15,15)
        v=mean+corr
        pred[j]=v.cpu().numpy()
        for k,ix in enumerate(j):
            if D.groups[ix]=='confirm':
                se[:,0]+=((v[k]-y[k]).square()*w[k]).sum((1,2),dtype=torch.float64).cpu().numpy()
                se[:,1]+=w[k].sum((1,2)).cpu().numpy();n+=1
        if (i//2)%160==0:print('maps',i,'time',round(time.monotonic()-t),flush=True)
    rmse=(se[:,0]/se[:,1])**.5
    ref=np.array(json.load(open(R/'audit/c79_receiver/blend_full/result.json'))['confirm']['rmse'])[:,0]
    assert n==1934 and np.max(abs(ref-rmse))<.00001,(ref,rmse)
    pred.flush();del pred;dest.replace(A/'base.npy')
    info.update(confirm_rmse=rmse.tolist(),elapsed=time.monotonic()-t)
    atomic_json(A/'base_index.json',info);print(rmse,flush=True)
if __name__=='__main__':main()
