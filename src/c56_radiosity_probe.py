"""Input-only coarse multi-bounce surface-power operator prototype; not ray-tracing GT."""
import sys,time,json,csv,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from torch.nn import functional as F
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.4)
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
M=np.load(R/'features/c49_visibility/train_margins.npy',mmap_mode='r')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
proto=json.loads((R/'audit/c43_architecture_20261102/protocol.json').read_text())
ids=proto['fit64_ids'][::max(1,len(proto['fit64_ids'])//8)][:8]
side=16;axis=(torch.arange(side,device='cuda',dtype=torch.float32)+.5)*256/side-.5
yy,xx=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack((xx,yy),-1).reshape(-1,2);N=len(xy);grid=(xy/127.5-1)[None,:,None]
rhos=torch.tensor([.2,.5,.8,.95],device='cuda');rec=[];t0=__import__('time').time()
with torch.inference_mode():
 for idx in ids:
  row=rows[idx];h=torch.tensor(np.array(H[int(row['hm_row'])]),device='cuda');sy,sx=torch.gradient(h,dim=(-2,-1))
  fields=torch.stack((h,sx,sy))[None];z,nx,ny=F.grid_sample(fields,grid,align_corners=True,padding_mode='border')[0,:,:,0]
  pos=torch.cat((xy,(z+.05)[:,None]),1);normal=torch.stack((-nx,-ny,torch.ones_like(nx)),1);norm=normal.norm(dim=1);normal/=norm[:,None]
  # K[receiver,emitter], with emitting patch area. Each row is capped for a stable open surface.
  vec=pos[None]-pos[:,None];dist=vec.norm(dim=-1).clamp_min(.1);unit=vec/dist[...,None]
  cr=(normal[:,None]*unit).sum(-1).clamp_min(0);ce=-(normal[None]*unit).sum(-1);ce=ce.clamp_min(0)
  margin=torch.full((N,N),-1e6,device='cuda')
  for lo in range(0,128,8):
   t=(torch.arange(lo,lo+8,device='cuda',dtype=torch.float32)+.5)/128
   points=pos[:,None,None]+vec[:,:,None]*t[None,None,:,None]
   sampled=F.grid_sample(h[None,None],(points[...,:2]/127.5-1).reshape(1,N*N,8,2),align_corners=True,padding_mode='border')[0,0].reshape(N,N,8)
   margin=torch.maximum(margin,(sampled-points[...,2]).amax(-1))
  vis=(margin<=0).float();K=256*norm[None]*cr*ce*vis/(math.pi*dist.square());K.fill_diagonal_(0)
  raw=K.sum(1);K*=torch.clamp(.95/raw.clamp_min(1e-9),max=1)[:,None]
  tr=int(row['tx_row']);tc=int(row['tx_col']);ht=h[tr,tc];frac=torch.remainder(ht,1)
  em=torch.tensor(np.array(M[idx],np.float32),device='cuda').permute(2,0,1)[None]
  em=F.grid_sample(em,grid,align_corners=True,padding_mode='border')[0,:,:,0].T
  for height,label in [(3.,'nominal3'),(3-float(frac),'raster_phase_candidate')]:
   tx=torch.stack((torch.tensor(float(tc),device='cuda'),torch.tensor(float(tr),device='cuda'),ht+height))
   v=tx[None]-pos;d=v.norm(dim=-1).clamp_min(.25);ci=(normal*v/d[:,None]).sum(-1).clamp_min(0)
   f=3-height;tv=torch.sigmoid(-((1-f)*em[:,0]+f*em[:,1])/.03)
   E=tv*ci/(4*math.pi*d.square());B=rhos[:,None]*E[None]
   for _ in range(16):B=rhos[:,None]*(E[None]+B@K.T)
   ratio=B/(rhos[:,None]*E[None]+1e-30)
   shadow=tv<.01
   rec.append({'map':int(idx),'height':label,'K_row_sum_mean':float(raw.mean()),'K_row_sum_max':float(raw.max()),
     'rows_capped_fraction':float((raw>.95).float().mean()),'shadow_nodes':int(shadow.sum()),
     'rho08_multi_over_single_gain_db_median':float((10*ratio[2].clamp_min(1).log10()).median()),
     'rho08_shadow_gain_db_median':float((10*ratio[2,shadow].clamp_min(1).log10()).median()) if shadow.any() else None,
     'power_min_max':[float(B.min()),float(B.max())]})
A=R/'audit/c56_multibounce_20261102';A.mkdir(exist_ok=True);out={'gpu':gpu,'side':side,'orders':16,'records':rec,'seconds':__import__('time').time()-t0,
 'input_only':True,'targets_masks_read':False,'limits':'Coarse16x16quadrature and row cap are candidate operators, not recovered simulator mesh or exact RT.'}
atomic_json(A/'probe.json',out);print(json.dumps(out),flush=True)
