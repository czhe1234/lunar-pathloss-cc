import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from c44_model import Model
torch.set_num_threads(8)
a=R/'audit/c44_dense_profile_20261102';p=json.loads((a/'protocol.json').read_text())
run=R/'runs/c44_profile_fit64';ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
net=Model('path');net.load_state_dict(ck['model']);net.eval()
import csv
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
i=p['monitor_fit_ids'][0];row=rows[i];H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
h=torch.from_numpy(np.array(H[int(row['hm_row'])],np.float32))[None];r=int(row['tx_row']);c=int(row['tx_col']);ht=h[0,r,c]
yy,xx=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
dx=xx[None]-c;dy=yy[None]-r;dz=h+1-ht-3;distance=torch.sqrt(dx*dx+dy*dy+dz*dz).clamp_min(.25)
sy,sx=torch.gradient(h,dim=(-2,-1));hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0]
x=torch.stack((h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(distance)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx*dx+dy*dy)/8)),1)
tx=torch.stack((ht*0+c,ht*0+r,ht+3))[None];pix=torch.tensor(p['monitor_pixels'][:512])[None]
with torch.inference_mode():
 state=net.encode(x,tx);local=state['local'].flatten(2).gather(2,pix[:,None].expand(-1,96,-1)).transpose(1,2)
 rz=h.flatten(1).gather(1,pix)+1
 prof=net.profile(x,tx,net.xy[pix],rz)
 normal=net.decode(state,pix)
 net.profile.out.weight.zero_();net.profile.out.bias.zero_();zero=net.decode(state,pix)
out={'step':ck['step'],'index':i,'local_feature_RMS':float(local.square().mean().sqrt()),
     'profile_feature_RMS':float(prof.square().mean().sqrt()),
     'profile_zero_mean_abs_delta_db':((normal-zero).abs().mean((0,1))*20).tolist(),
     'scope':'CPU frozen-model internal branch-use test on1 preselected FIT map,512 fixed pixels; no labels; not a physical-path explanation'}
(a/f'branch_check_step{ck["step"]}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
