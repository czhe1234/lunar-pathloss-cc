"""No training: expose unclipped native powers, then change diffuse integration density only."""
import sys,json,csv,time,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c69_registered import RegisteredTransport
import torch,numpy as np
A=R/'audit/c121_pairs'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dirs',type=int,default=128);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(3)
 proto=json.load(open(A/'power_probe_protocol.json'));ids=proto['maps'];out=A/f'power{a.dirs}';out.mkdir(exist_ok=False)
 engine=RegisteredTransport(physical=1,anti=0,smooth=1,direct=2,spec=4,diffuse=a.dirs)
 rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));hh=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
 scene=engine.xy[engine.faces].mean(1);start=time.monotonic();max_lowclip=0.
 # Pure H/TX inference: no target or mask files opened.
 for j,i in enumerate(ids):
  r=rows[i];h=hh[int(r['hm_row'])];tr=int(r['tx_row']);tc=int(r['tx_col'])
  direct,spec,diff,ht,x,y,z,emit=engine.raw(h,tr,tc,return_emission=True)
  alt=ht.reshape(-1)[engine.faces].mean(1)+1;d2=((scene[:,0]-x*engine.spacing)**2+(scene[:,1]-y*engine.spacing)**2+(alt-z)**2).clamp_min(.0625)
  powers=torch.cat([direct[:,:2],spec[:,:2],diff[:,:2],direct[:,:2]+spec[:,:2]+diff[:,:2],emit[:,:2]],1)*d2[:,None]
  values=powers.cpu().numpy();np.save(out/f'{i}.npy',values)
  print(a.dirs,j+1,len(ids),round(time.monotonic()-start,2),flush=True)
 atomic_json(out/'done.json',dict(maps=ids,dirs=a.dirs,gpu=g,seconds=time.monotonic()-start,labels_read=False,mesh_mapping_changed=False))
if __name__=='__main__':main()
