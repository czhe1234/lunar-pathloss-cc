"""Frozen C82 FIT predictions from input-only CC caches. No label/mask file opened."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c67_dense import model_inputs
from src.c72_score_dev import load_net
from src.c73_full import graph_net
from src.c3_geometry import basic_inputs
from src.c82_blend import field_features
A=R/'audit/c101_residual'
def mm(p):return np.load(p,mmap_mode='r')
def lookup(patterns,file):
 result={}
 for pattern in patterns:
  for d in R.glob(pattern):
   if not (d/'complete.json').exists() or not (d/file).exists():continue
   ar=mm(d/file);result.update({int(i):(ar,j) for j,i in enumerate(json.loads((d/'indices.json').read_text()))})
 return result
def gather(lk,ids):
 ar,p=lk[ids[0]]
 if all(lk[i][0] is ar and lk[i][1]==p+j for j,i in enumerate(ids)):return np.array(ar[p:p+len(ids)])
 return np.stack([lk[i][0][lk[i][1]] for i in ids])
class Predictor:
 def __init__(self):
  self.rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));self.h=mm(R.parent/'data/train_hm.npy')
  self.g3=mm(R/'features/c3/train.npy');self.g4=mm(R/'features/c4_extra/train.npy')
  self.fitbase=lookup(['features/c66_frozen_maps/fit*of2','features/c75_fit_extension/fit*'],'prediction.npy')
  self.native=lookup(['features/c72_native/part*','features/c72_dev_extra_native/part*','features/c75_fit_extension/fit*'],'features.npy')
  self.compact=lookup(['features/c73_compact/part*','features/c73_dev_extra_compact/part*'],'features.npy')
  p=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1';ix=json.loads((p/'indices.json').read_text())['index_rows']
  self.devindex={int(i):j for j,i in enumerate(ix)};self.devbase=[mm(p/f'pl{b}.npy') for b in ['415','58']]
  self.reference=mm(R/'audit/c88_mesh/base.npy')
  self.local=load_net(R/'runs/c72_observed');self.graph=graph_net('c73_wide_real')
  cfg=json.loads((R/'runs/c82_selected/spectrum.json').read_text());self.mode=cfg['mode'];self.beta=[torch.tensor(x,device='cuda') for x in cfg['beta']]
 @torch.inference_mode()
 def predict(self,ids):
  to=lambda x:torch.from_numpy(np.ascontiguousarray(x)).cuda()
  rows=[self.rows[i] for i in ids]
  h=to(self.h[[int(r['hm_row']) for r in rows]]).float();tr=to(np.array([int(r['tx_row']) for r in rows]));tc=to(np.array([int(r['tx_col']) for r in rows]))
  if ids[0] in self.fitbase:base=to(gather(self.fitbase,ids)).float()
  else:base=to(np.stack([p[[self.devindex[i] for i in ids]] for p in self.devbase],1)).float()
  native=to(gather(self.native,ids)).float();compact=to(gather(self.compact,ids)).float()
  # Reproduce Panel storage precision before the original C72 network.
  local_input=model_inputs(h,tr,tc,base,to(self.g3[ids]),to(self.g4[ids]),native).half().float().contiguous(memory_format=torch.channels_last)
  basic,fs=basic_inputs(h,tr,tc);prior=fs+(-20*(compact[:,31:33]+compact[:,41:43])).clamp(-8,48)
  graph_input=torch.cat([basic,compact],1).contiguous(memory_format=torch.channels_last)
  with torch.autocast('cuda',torch.bfloat16):
   correction=self.local(local_input).float().half().float()
   graph=prior+self.graph(graph_input,h,tr,tc).float()
  local=base+correction
  yy,xx=torch.meshgrid(torch.arange(256,device='cuda'),torch.arange(256,device='cuda'),indexing='ij')
  z=h[torch.arange(len(ids),device='cuda'),(tr+1).clamp_max(255),tc].floor()+3
  distance=torch.sqrt((xx-tc[:,None,None]).square()+(yy-tr[:,None,None]).square()+(h+1-z[:,None,None]).square()).clamp_min(.25)
  fs=20*torch.log10(distance[:,None]*torch.tensor([415e6,5.775e9],device='cuda')[None,:,None,None]*(4*np.pi/299792458.))
  vis=compact[:,16:17]
  delta=torch.cat([(field_features(base,local,graph,fs,vis,self.mode,b)*self.beta[b][None,:,None,None]).sum(1,keepdim=True) for b in [0,1]],1).clamp(-15,15)
  return .5*(local+graph)+delta
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--of',type=int,default=1);ap.add_argument('--check',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 A.mkdir(exist_ok=True);net=Predictor()
 if a.check:
  dm=json.loads((R/'audit/c96_information/dev/done.json').read_text())
  ids=[int(i) for i,g in zip(dm['maps'],dm['groups']) if g!='confirm'][:8]
  pred=net.predict(ids).cpu().numpy();ref=np.stack([net.reference[net.devindex[i]] for i in ids]);diff=pred-ref
  rec=dict(ids=ids,point_rms=np.sqrt((diff.astype('float64')**2).mean((0,2,3))).tolist(),max_abs=float(abs(diff).max()),GPU=gpu)
  # Cross-device BF16 replay can differ. A bad input mapping must not pass.
  assert max(rec['point_rms'])<.025,rec
  atomic_json(A/'replay.json',rec);print(json.dumps(rec),flush=True);return
 assert (A/'replay.json').exists()
 ids=[]
 for d in sorted((R/'audit/c90_facet').glob('fit[0-9]')):ids+=json.loads((d/'done.json').read_text())['ids']
 assert len(set(ids))==len(ids)==2815
 ids=sorted(ids,key=lambda i:(str(net.native[i][0].filename),net.native[i][1]))[a.shard::a.of]
 out=A/f'base{a.shard}of{a.of}';out.mkdir(exist_ok=False);atomic_json(out/'indices.json',ids)
 ar=np.lib.format.open_memmap(out/'prediction.npy',mode='w+',shape=(len(ids),2,256,256),dtype='float32')
 t=time.monotonic()
 for start in range(0,len(ids),8):
  ix=ids[start:start+8];v=net.predict(ix).cpu().numpy();assert np.isfinite(v).all()
  ar[start:start+len(ix)]=v
  if start%128==0 or start+len(ix)==len(ids):
   rec=dict(maps=start+len(ix),total=len(ids),seconds=time.monotonic()-t);atomic_json(out/'progress.json',rec);print(rec,flush=True)
 ar.flush();atomic_json(out/'done.json',dict(maps=len(ids),GPU=gpu,seconds=time.monotonic()-t,input_only=True,labels_masks_opened=False,source='Frozen development C82 with C72half-correction,C73graph and fixedC82spectrum.'))
if __name__=='__main__':main()
