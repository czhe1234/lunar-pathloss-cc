"""Freeze a mixture using the monitor split. Quantify transfer and compare physical vs shuffled graph edges."""
import argparse,csv,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import DATA,gpu_guard,atomic_json
from src.c73_train import TerrainData
from src.c73_graph import GraphField
import torch
import numpy as np

def graph_net(name):
    ck=torch.load(R/'runs'/name/'best.pt',map_location='cpu',weights_only=True)
    net=GraphField(56,ck['width'],ck['shuffle']).cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(ck['state']);return net
from src.c74_mixture_score import summarize
@torch.inference_mode()
def main():
    gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
    out=R/'runs/c73_wide_real/full_internal';out.mkdir(exist_ok=False)
    pr=R/'runs/c72_observed/full_internal';ids=json.load(open(pr/'indices.json'))
    base_path=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1';assert json.load(open(base_path/'indices.json'))['index_rows']==ids
    base=[np.load(base_path/f'pl{b}.npy',mmap_mode='r') for b in ('415','58')];cor=np.load(pr/'correction.npy',mmap_mode='r')
    atomic_json(out/'indices.json',ids);nets=[graph_net(n) for n in ['c73_wide_real','c73_wide_shuffled']]
    arr=np.lib.format.open_memmap(out/'prediction.npy','w+',dtype=np.float32,shape=(len(ids),2,256,256))
    sums=[];controls=[];start=time.monotonic()
    for j in range(0,len(ids),96):
        data=TerrainData(ids[j:j+96],['features/c73_compact','features/c73_dev_extra_compact'])
        for k in range(0,len(data.ids),8):
            ix=torch.arange(k,min(k+8,len(data.ids)),device='cuda');sl=slice(j+k,j+k+len(ix))
            x,h,tr,tc,p0,y,w=data.batch(ix)
            with torch.autocast('cuda',torch.bfloat16):g=p0+nets[0](x,h,tr,tc).float();ctrl=p0+nets[1](x,h,tr,tc).float()
            b=torch.from_numpy(np.stack([m[sl] for m in base],axis=1)).cuda().float()
            local=b+torch.from_numpy(np.array(cor[sl],copy=True)).cuda().float();arr[sl]=g.cpu().numpy()
            # The first 11 channels are basic terrain inputs. The rest are compact geometry and power features.
            v=x[:,11+16:11+17];z=torch.where(v>.99,0,torch.where(v>=.01,1,torch.where(x[:,11+41:11+43]<-.8,3,2)))
            se=[(b-y).square(),(local-y).square(),(g-y).square(),(g-local).square(),torch.ones_like(y)]
            batch=[]
            for zz in range(4):
                m=w&(z==zz);batch.append(torch.stack([(q*m).sum((2,3),dtype=torch.float64) for q in se],dim=-1))
            sums.append(torch.stack(batch,dim=2).cpu().numpy());controls.append(((ctrl-y).square()*w).sum((2,3),dtype=torch.float64).cpu().numpy())
        if j%384==0 or j+len(data.ids)==len(ids):print('maps',j+len(data.ids),len(ids),'seconds',round(time.monotonic()-start,1),flush=True)
        del data
    arr.flush();sums=np.concatenate(sums);controls=np.concatenate(controls);np.savez(out/'sse_by_zone.npz',statistics=sums,shuffled=controls)
    summarize(ids,sums,controls,out)
if __name__=='__main__':main()
