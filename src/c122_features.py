"""Targeted recovery of clipped / angularly undersampled physical powers. No labels opened."""
import sys,json,csv,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c69_registered import RegisteredTransport
import torch,numpy as np
A=R/'audit/c122_power'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(3);out=A/f'part{a.shard}';out.mkdir(exist_ok=False)
 records=json.load(open(A/'protocol.json'))['records'][a.shard::3]
 engine={n:RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=n) for n in [128,8192]}
 look=np.load(R/'src/c70_centroid_lookup0.npz');ids=look['ids'];wt=look['weights']
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));hh=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 selected={};pixels={};mapids={}
 for part in ['fit','monitor']:
  selected[part]=np.load(R/f'audit/c121_pairs/tail_{part}/rows.npy');pixels[part]=np.load(R/f'audit/c101_residual/{part}/pixel.npy',mmap_mode='r');mapids[part]=np.load(R/f'audit/c101_residual/{part}/mapid.npy',mmap_mode='r')[selected[part]]
 scene=engine[128].xy[engine[128].faces].mean(1);start=time.monotonic()
 for j,rec in enumerate(records):
  part=rec['part'];i=rec['mapid'];ix=np.flatnonzero(mapids[part]==i);px=np.array(pixels[part][selected[part][ix]])
  order=np.argsort(-wt[px],axis=1,kind='stable');pick=np.take_along_axis(ids[px],order,1);w=np.take_along_axis(wt[px],order,1)
  cell=pick%65025;face=pick//65025;rr,cc=np.divmod(cell,255);pr,pc=np.divmod(px,256)
  meta=np.c_[w,cc+np.where(face==0,2/3,1/3)-pc[:,None],rr+np.where(face==0,1/3,2/3)-pr[:,None]]
  r=rows[i];h=hh[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col']);values={}
  for n,eng in engine.items():
   direct,spec,diff,ht,x,y,z,emit=eng.raw(h,tr,tc,return_emission=True)
   alt=ht.reshape(-1)[eng.faces].mean(1)+1;d2=((scene[:,0]-x*eng.spacing)**2+(scene[:,1]-y*eng.spacing)**2+(alt-z)**2).clamp_min(.0625)
   power=torch.cat([direct[:,:2],spec[:,:2],diff[:,:2],direct[:,:2]+spec[:,:2]+diff[:,:2],emit[:,:2]],1)*d2[:,None]
   raw=power.cpu().numpy()[pick].astype('float64')
   for key,floor in ([('clip128',1e-12),('wide128',1e-38)] if n==128 else [('wide8192',1e-38)]):
    p=raw.clip(floor,1e8);log=np.log10(p)/4
    values[key]=np.c_[log.reshape(len(ix),30),(log*w[:,:,None]).sum(1),np.log10((p*w[:,:,None]).sum(1))/4,meta].astype('float32')
  np.savez(out/f'{part}_{i}.npz',index=ix,**values)
  if (j+1)%50==0:print('shard',a.shard,j+1,len(records),'seconds',round(time.monotonic()-start,1),flush=True)
 atomic_json(out/'done.json',dict(shard=a.shard,records=len(records),GPU=g,seconds=time.monotonic()-start,labels_opened=False))
if __name__=='__main__':main()
