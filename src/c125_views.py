"""Input-only native H/TX views for already frozen diagnostic point lists."""
import sys,csv,json,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from torch.nn import functional as F
A=R/'audit/c125_tail';S=R/'audit/c101_residual';T=R/'audit/c120_sign'
@torch.no_grad()
def views(h,tr,tc,pixel):
 device=h.device;px=torch.as_tensor(pixel,device=device,dtype=torch.long);rr=(px//256).float();cc=(px%256).float();n=len(px)
 def patch(r,c,size):
  offset=torch.arange(-(size//2),size//2+1,device=device)
  y=(r[:,None,None]+offset[None,:,None]).long().clamp(0,255)
  x=(c[:,None,None]+offset[None,None,:]).long().clamp(0,255)
  hh=h[y,x];r0=r.long();c0=c.long();center=h[r0,c0]
  gx=(h[r0,(c0+1).clamp_max(255)]-h[r0,(c0-1).clamp_min(0)])/2
  gy=(h[(r0+1).clamp_max(255),c0]-h[(r0-1).clamp_min(0),c0])/2
  # First view retains raw relative height. Second exposes centimetre-scale nonplanarity.
  plane=gx[:,None,None]*(x-c0[:,None,None])+gy[:,None,None]*(y-r0[:,None,None])
  return torch.stack([(hh-center[:,None,None])/5,(hh-center[:,None,None]-plane)/.1],1)
 rx=patch(rr,cc,33)
 tx=patch(torch.tensor([float(tr)],device=device),torch.tensor([float(tc)],device=device),17).expand(n,-1,-1,-1)
 t=torch.linspace(0,1,129,device=device);width=torch.tensor([-16,-8,-4,-2,0,2,4,8,16],device=device,dtype=torch.float32)
 dr=rr-tr;dc=cc-tc;d=(dr.square()+dc.square()).sqrt().clamp_min(1)
 ys=tr+dr[:,None,None]*t[None,:,None]+(dc/d)[:,None,None]*width[None,None,:]
 xs=tc+dc[:,None,None]*t[None,:,None]-(dr/d)[:,None,None]*width[None,None,:]
 grid=torch.stack([xs/127.5-1,ys/127.5-1],-1).reshape(1,n*129,9,2)
 hp=F.grid_sample(h[None,None],grid,mode='bilinear',padding_mode='border',align_corners=True).reshape(n,129,9)
 line=(h[tr,tc]+3)*(1-t)[None,:,None]+(h[rr.long(),cc.long()]+1)[:,None,None]*t[None,:,None]
 corridor=((hp-line)/5).transpose(1,2)
 return rx,tx,corridor
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['fit','monitor'],required=True);a=ap.parse_args();g=gpu_guard();torch.set_num_threads(3)
 part=a.part;P=S/part;out=A/'views'/part;out.mkdir(parents=True,exist_ok=False)
 ids=np.load(T/(part+'_rows.npy'));mapid=np.load(P/'mapid.npy',mmap_mode='r')[ids];pixel=np.load(P/'pixel.npy',mmap_mode='r')[ids]
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));height=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 arrays={key:np.lib.format.open_memmap(out/(key+'.npy'),'w+',dtype=np.float32,shape=(len(ids),)+shape) for key,shape in [('rx',(2,33,33)),('tx',(2,17,17)),('corridor',(9,129))]}
 t0=time.monotonic();maps=np.unique(mapid);maxerr=0.
 for j,mid in enumerate(maps):
  ix=np.flatnonzero(mapid==mid);r=rows[int(mid)];h=torch.tensor(np.array(height[int(r['hm_row'])]),device='cuda',dtype=torch.float32)
  v=views(h,int(r['tx_row']),int(r['tx_col']),pixel[ix])
  for key,val in zip(['rx','tx','corridor'],v):
   x=val.cpu().numpy();assert np.isfinite(x).all();arrays[key][ix]=x
  # Exact centre height subtraction; explicit receiver patch has no target/mask input.
  maxerr=max(maxerr,float(v[0][:,0,16,16].abs().max()))
  if j%128==0 or j+1==len(maps):
   rec=dict(maps=j+1,total=len(maps),points=len(ids),seconds=time.monotonic()-t0);atomic_json(out/'progress.json',rec);print(rec,flush=True)
 assert maxerr==0
 for x in arrays.values():x.flush()
 np.save(out/'rows.npy',ids)
 atomic_json(out/'done.json',dict(status='complete',GPU=g,part=part,points=len(ids),maps=len(maps),seconds=time.monotonic()-t0,labels_read=False,masks_read=False,coordinates='published raster row/col, representation only; not a candidate mesh remapping',dtype='float32',receiver_center_error=maxerr))
if __name__=='__main__':main()
