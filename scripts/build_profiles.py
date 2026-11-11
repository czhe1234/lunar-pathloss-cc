"""Input-only straight-line terrain features. No pathloss/mask files or propagation simulator."""
import sys,fcntl,time,csv,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,GPU_UUID,gpu_guard,atomic_json
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val'],default='train');p.add_argument('--samples',type=int,default=192);p.add_argument('--batch',type=int,default=8);args=p.parse_args()
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();torch.set_num_threads(8);out=ROOT/'features';out.mkdir(exist_ok=True)
assert args.samples==192, 'Training and inference currently use 192 samples'
assert not (out/f'{args.split}_profile.pt').exists(), 'Refuse to overwrite an existing feature cache'
rows=list(csv.DictReader((DATA/f'{args.split}_index.csv').open()))
hm=np.load(DATA/f'{args.split}_hm.npy',mmap_mode='r',allow_pickle=False);h=torch.from_numpy(np.array(hm,copy=True)).cuda()
yy,xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij');yy=yy.flatten();xx=xx.flatten()
t=torch.linspace(0,1,args.samples+2,device='cuda')[1:-1]
result=torch.empty((len(rows),2,256,256),dtype=torch.float16,device='cpu')
t0=time.time()
with torch.inference_mode():
 for lo in tqdm(range(0,len(rows),args.batch),desc=f'{args.split} input terrain profiles',mininterval=2.):
  rr=rows[lo:lo+args.batch];ids=torch.tensor([int(r['hm_row']) for r in rr],device='cuda');tr=torch.tensor([int(r['tx_row']) for r in rr],device='cuda');tc=torch.tensor([int(r['tx_col']) for r in rr],device='cuda');a=h[ids];b=len(rr)
  px=tc[:,None,None]+(xx[None,:,None]-tc[:,None,None])*t[None,None,:]
  py=tr[:,None,None]+(yy[None,:,None]-tr[:,None,None])*t[None,None,:]
  grid=torch.stack((px/127.5-1.,py/127.5-1.),dim=-1)
  terrain=F.grid_sample(a[:,None],grid,mode='bilinear',padding_mode='border',align_corners=True)[:,0]
  txh=a[torch.arange(b,device='cuda'),tr,tc]+3.
  receiver=a.flatten(1)+1.
  straight_height=txh[:,None,None]+(receiver[:,:,None]-txh[:,None,None])*t[None,None,:]
  excess=terrain-straight_height
  features=torch.stack((excess.amax(dim=-1),excess.clamp_min(0).mean(dim=-1)),dim=1).reshape(b,2,256,256)/100.
  assert torch.isfinite(features).all()
  result[lo:lo+b]=features.to(device='cpu',dtype=torch.float16)
path=out/f'{args.split}_profile.pt';tmp=out/f'{args.split}_profile.tmp';torch.save(result,tmp);tmp.replace(path)
source=json.loads((ROOT/'audit/raw_audit.json').read_text())['files']
atomic_json(out/f'{args.split}_profile.json',{'split':args.split,'samples_on_segment':args.samples,'endpoint_excluded':True,'features':['max_terrain_minus_straight_height_m/100','mean_positive_terrain_minus_straight_height_m/100'],'tx_height_m':3,'rx_height_m':1,'inputs':{k:source[k] for k in [f'{args.split}_index.csv',f'{args.split}_hm.npy']},'no_targets_or_validity_masks_used':True,'shape':list(result.shape),'seconds':time.time()-t0,'device':info})
print('PROFILE_COMPLETE',path,time.time()-t0,flush=True)
