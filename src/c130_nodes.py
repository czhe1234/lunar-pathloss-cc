"""Raw H/TX-only critical-obstruction nodes and oriented 2D terrain patches."""
import sys,json,csv,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from torch.nn import functional as F
A=R/'audit/c130_projective';P=R/'audit/c101_residual';T=R/'audit/c120_sign'
@torch.no_grad()
def nodes(h,tr,tc,pixels):
 h=h.float()
 px=torch.as_tensor(pixels,device=h.device,dtype=torch.long);rr=(px//256).float();cc=(px%256).float();N=len(px)
 dr=rr-tr;dc=cc-tc;distance=(dr.square()+dc.square()).sqrt().clamp_min(.5);ur=dr/distance;uc=dc/distance
 tx=h[tr,tc]+3;rx=h[rr.long(),cc.long()]+1;t=torch.arange(1,512,device=h.device)/512
 def sample(fields,r,c):
  shape=r.shape;grid=torch.stack([c/127.5-1,r/127.5-1],-1).reshape(1,-1,1,2)
  return F.grid_sample(fields[None],grid,padding_mode='border',align_corners=True).reshape(fields.shape[0],*shape)
 hp=sample(h[None],tr+dr[:,None]*t,tc+dc[:,None]*t)[0]
 def select(lo,hi,zlo,zhi):
  frac=(t[None]-lo[:,None])/(hi-lo).clamp_min(1/512)[:,None]
  ok=(frac>1e-4)&(frac<1-1e-4)
  clearance=hp-(zlo[:,None]*(1-frac)+zhi[:,None]*frac)
  fresnel=(distance[:,None]*(hi-lo)[:,None]*frac*(1-frac)).clamp_min(1e-8).sqrt()
  score=(clearance/fresnel).masked_fill(~ok,-1e8);ii=score.argmax(1);valid=ok.any(1);ar=torch.arange(N,device=h.device)
  return t[ii],hp[ar,ii],clearance[ar,ii],score[ar,ii].masked_fill(~valid,0),valid
 zero=torch.zeros(N,device=h.device);one=zero+1;zz=zero+tx
 root=select(zero,one,zz,rx);left=select(zero,root[0],zz,root[1]);right=select(root[0],one,root[1],rx)
 sy,sx=torch.gradient(h);fields=torch.stack([h,sy,sx])
 patches=[];metas=[]
 offset=torch.arange(-8,9,device=h.device);along=offset[None,:,None];across=offset[None,None,:]
 for node in [root,left,right]:
  at,hc,clear,score,ok=node;nr=tr+dr*at;nc=tc+dc*at
  center=sample(fields,nr,nc);grad_r=center[1];grad_c=center[2]
  channels=[]
  for scale in [1.,4.]:
   ro=scale*(ur[:,None,None]*along+uc[:,None,None]*across)
   co=scale*(uc[:,None,None]*along-ur[:,None,None]*across)
   hh=sample(h[None],nr[:,None,None]+ro,nc[:,None,None]+co)[0]
   plane=grad_r[:,None,None]*ro+grad_c[:,None,None]*co
   channels.extend([torch.asinh((hh-center[0,:,None,None])/(2*scale)),torch.asinh((hh-center[0,:,None,None]-plane)/(.05*scale))])
  patches.append(torch.stack(channels,1)*ok[:,None,None,None])
  metas.append(torch.stack([at,torch.asinh(clear/2)/3,torch.asinh(score)/3,distance*at/256,distance*(1-at)/256,
   torch.asinh(grad_r*ur+grad_c*uc),torch.asinh(grad_r*uc-grad_c*ur),ok.float()],1))
 return torch.stack(patches,1),torch.stack(metas,1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['fit','monitor']);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(3)
 if a.check:
  h=torch.zeros((256,256),device='cuda');x,m=nodes(h,100,64,[100*256+192])
  assert x.abs().max()==0 and abs(m[0,0,0].item()-.75)<.005
  h=(12-(torch.arange(256,device='cuda')-128).abs()).clamp_min(0)[None].expand(256,256).clone()
  x,m=nodes(h,100,64,[100*256+192]);assert abs(m[0,0,0].item()-.5)<.005
  hr=torch.rot90(h,1,[0,1]);xr,mr=nodes(hr,191,100,[(255-192)*256+100])
  assert (x-xr).abs().max()<1e-3 and (m-mr).abs().max()<1e-4
  atomic_json(A/'node_geometry_check.json',dict(status='passed',GPU=gpu,flat_peak_fraction=.75,ridge_peak_fraction=.5,rotation_patch_max_error=float((x-xr).abs().max()),labels_read=False,masks_read=False));print('CHECK PASSED');return
 selected=np.load(T/(a.part+'_rows.npy'));mi=np.load(P/a.part/'mapid.npy',mmap_mode='r')[selected];px=np.load(P/a.part/'pixel.npy',mmap_mode='r')[selected]
 rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));hm=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 out=A/'nodes'/a.part;out.mkdir(parents=True,exist_ok=False)
 patch=np.lib.format.open_memmap(out/'patch.npy','w+',dtype='float32',shape=(len(selected),3,4,17,17))
 meta=np.lib.format.open_memmap(out/'meta.npy','w+',dtype='float32',shape=(len(selected),3,8));start=time.monotonic();maps=np.unique(mi)
 for k,i in enumerate(maps):
  at=np.flatnonzero(mi==i);row=rows[int(i)];h=torch.tensor(np.array(hm[int(row['hm_row'])]),device='cuda',dtype=torch.float32)
  for st in range(0,len(at),512):
   sl=at[st:st+512];x,m=nodes(h,int(row['tx_row']),int(row['tx_col']),px[sl]);xn=x.cpu().numpy();mn=m.cpu().numpy()
   assert np.isfinite(xn).all() and np.isfinite(mn).all();patch[sl]=xn;meta[sl]=mn
  if k%128==0 or k+1==len(maps):
   z=dict(maps=k+1,total=len(maps),seconds=time.monotonic()-start);atomic_json(out/'progress.json',z);print(z,flush=True)
 patch.flush();meta.flush();np.save(out/'rows.npy',selected)
 atomic_json(out/'done.json',dict(status='complete',part=a.part,points=len(selected),maps=len(maps),seconds=time.monotonic()-start,GPU=gpu,labels_read=False,masks_read=False,
  source='publishedH/TX,512samplegeometry,3recursivebottlenecks,rawrotated17x17patches at1m/4m; no input from otherTXlabels'))
if __name__=='__main__':main()
