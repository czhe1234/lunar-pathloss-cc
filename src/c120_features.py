import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c118_location import AuxField
from src.c105_context import reader,ResidualPanel
A=R/'audit/c120_sign'
@torch.inference_mode()
def main():
 g=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');path=R/'runs/c118_location/best.pt'
 ck=torch.load(path,map_location='cpu',weights_only=False);net=AuxField().cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);net.eval()
 hidden={}
 hook=net.head.register_forward_pre_hook(lambda module,args:hidden.update(v=args[0]))
 r=reader();start=time.monotonic()
 for part in ['fit','monitor']:
  P=R/'audit/c101_residual'/part;meta=json.loads((P/'done.json').read_text());ids=meta['ids'];pp=meta['points_per_map']
  y=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r')
  tail=((np.abs(y[:,:2]-base)>=5)&(y[:,2:4]>0));selected=np.flatnonzero(tail.any(1));lookup=np.full(len(y),-1,np.int64);lookup[selected]=np.arange(len(selected))
  np.save(A/(part+'_rows.npy'),selected)
  arr=np.lib.format.open_memmap(A/(part+'_native.npy'),'w+',dtype='float32',shape=(len(selected),96))
  d=ResidualPanel(r,ids,'context');pixel=torch.tensor(np.load(P/'pixel.npy').reshape(len(ids),pp),device='cuda',dtype=torch.int64)
  for st in range(0,len(ids),16):
   en=min(st+16,len(ids));sl=slice(st,en);x,b,yy,w=d.batch(sl)
   with torch.autocast('cuda',torch.bfloat16):net(x)
   z=hidden['v'];v=z.flatten(2).gather(2,pixel[sl,None].expand(-1,96,-1)).transpose(1,2).reshape(-1,96).float().cpu().numpy()
   lo=st*pp;hi=en*pp;keep=np.flatnonzero(lookup[lo:hi]>=0);arr[lookup[lo:hi][keep]]=v[keep]
  arr.flush();atomic_json(A/(part+'_features.json'),dict(points=len(y),selected_union_true_tail_rows=len(selected),features=96,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epoch=ck['epoch'],GPU=g,seconds=time.monotonic()-start,scope='TRUE_TAIL_ORACLE_GATE diagnostic;features remain input-only;not deployable powerprediction'))
  print('FEATURES',part,len(selected),flush=True);del d;torch.cuda.empty_cache()
 hook.remove()
if __name__=='__main__':main()
