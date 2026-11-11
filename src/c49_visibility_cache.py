"""Input-only TX->terrain visibility margins; never opens PL or validity masks."""
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.35)
O=R/'features/c49_visibility';O.mkdir(exist_ok=False)
src=R/'features/c3/train.npy';g=np.load(src,mmap_mode='r');out=np.lib.format.open_memmap(O/'train_margins.npy',mode='w+',dtype=np.float16,shape=(len(g),256,256,2))
t=(torch.arange(8,device='cuda',dtype=torch.float32)+.5)/8
start=time.time()
with torch.inference_mode():
 for lo in range(0,len(g),32):
  z=torch.from_numpy(np.array(g[lo:lo+32,:8],copy=True)).cuda().float()
  z=2*torch.sinh(z*3)
  em3=(z+.95*t[None,:,None,None]).amax(1)
  em2=(z+.95*t[None,:,None,None]+(1-t)[None,:,None,None]).amax(1)
  value=torch.stack((em3,em2),-1)
  assert torch.isfinite(value).all()
  out[lo:lo+len(z)]=value.half().cpu().numpy()
  if lo%1024==0:
   p={'maps':min(lo+32,len(g)),'total':len(g),'seconds':time.time()-start};atomic_json(O/'progress.json',p);print(json.dumps(p),flush=True)
out.flush()
meta={'complete':True,'maps':len(g),'source':str(src),'shape':list(out.shape),'input_only':True,'PL_or_validity_masks_read':False,
 'formula':'ExistingC7 margin: max8bin[2sinh(3g)+.95t+(3-TXheight)(1-t)], TXheight3/2',
 'interpretation':'TX visibility to terrain+.05;8bin midpoint proxy; height3 uncertainty<=.059375m, not actual path visibility',
 'gpu':gpu,'seconds':time.time()-start}
atomic_json(O/'meta.json',meta);print(json.dumps(meta),flush=True)
