import sys,os,csv,json,time,argparse,fcntl,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,atomic_json,check_source_unchanged
from c4_geometry import extra_geometry as geometry
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--split',default='train',choices=['train','val']);p.add_argument('--batch',type=int,default=4);args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();check_source_unchanged();torch.set_num_threads(8)
rows=list(csv.DictReader((DATA/f'{args.split}_index.csv').open()));hm=np.load(DATA/f'{args.split}_hm.npy',mmap_mode='r')
out=ROOT/'features/c4_extra';out.mkdir(parents=True,exist_ok=True);name=out/f'{args.split}.npy';state=out/f'{args.split}.json'
sha=hashlib.sha256((ROOT/'src/c4_geometry.py').read_bytes()).hexdigest();start=0
if state.exists():
 old=json.loads(state.read_text());assert old['source_sha256']==sha
 if old['complete']:print('cache already complete');sys.exit()
 start=old['done']
arr=np.lib.format.open_memmap(name,mode='r+' if name.exists() else 'w+',dtype=np.float16,shape=(len(rows),4,256,256))
t0=time.time()
for lo in range(start,len(rows),args.batch):
 rr=rows[lo:lo+args.batch]
 h=torch.from_numpy(np.array(hm[[int(a['hm_row']) for a in rr]],copy=True)).cuda()
 r=torch.tensor([int(a['tx_row']) for a in rr],device='cuda');c=torch.tensor([int(a['tx_col']) for a in rr],device='cuda')
 with torch.inference_mode():g=geometry(h,r,c).cpu().numpy()
 assert np.isfinite(g).all();arr[lo:lo+len(rr)]=g
 if (lo-start)//args.batch%100==0 or lo+len(rr)==len(rows):
  arr.flush();s={'split':args.split,'done':lo+len(rr),'total':len(rows),'complete':lo+len(rr)==len(rows),'elapsed_seconds':time.time()-t0,'maps_per_second':(lo+len(rr)-start)/(time.time()-t0),'source_sha256':sha,'device':info,'inputs_used':[f'{args.split}_hm.npy',f'{args.split}_index.csv']};atomic_json(state,s);print(json.dumps(s),flush=True)
