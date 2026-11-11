"""C61 public H/TX inference. No indices, labels, masks, or feature caches."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard
import numpy as np,torch
from torch.nn import functional as F
from src.c61_model import Model
from src.c47_inference import Predictor as Features
from src.c3_geometry import geometry
class InferenceModel(Model):
 def global_features(self,x):
  precision=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('high')
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   latent=self.backbone.embedding(x)[:,None].expand(-1,8,-1)
   for layer in self.backbone.layers:latent=F.relu(layer(latent))
   with torch.autocast('cuda',enabled=False):
    q=torch.einsum('nkw,kwo->nko',latent.float(),self.backbone.head)+self.backbone.bias
    mean=latent.float().mean(1);std=latent.float().var(1,unbiased=False).clamp_min(1e-8).sqrt()
    raw=((x.float()-self.embedding.mean)/self.embedding.std).clamp(-20,20)
    features=torch.cat((mean,std,raw,q.mean(1),q.var(1,unbiased=False).clamp_min(1e-8).sqrt()),1)
    self.last_base=q.mean(1)
  torch.set_float32_matmul_precision(precision)
  return self.global_blocks(F.silu(self.global_in(features)))
class Predictor(Features):
 def __init__(self,model_directory,precision='fp32',chunk=4096):
  self.gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.65);self.precision=precision;self.chunk=chunk
  d=Path(model_directory);ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config']
  state={k[len('backbone.'):]:v for k,v in ck['model'].items() if k.startswith('backbone.')}
  self.net=InferenceModel(state,cfg['variant']).cuda().eval();self.net.load_state_dict(ck['model'])
  yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
 @torch.inference_mode()
 def frame(self,H,tr,tc):
  h=torch.from_numpy(np.array(H,np.float32,copy=True)).cuda();sy,sx=torch.gradient(h,dim=(-2,-1));yy=torch.from_numpy(self.yy).cuda();xx=torch.from_numpy(self.xx).cuda()
  ht=h[tr,tc];dx=xx-tc;dy=yy-tr;dz=h+1-ht-3;d=torch.sqrt(dx.square()+dy.square()+dz.square()).clamp_min(.25)
  hp=h-F.avg_pool2d(F.pad(h[None,None],(2,2,2,2),mode='replicate'),5,stride=1)[0,0]
  g=geometry(h[None],torch.tensor([tr],device='cuda'),torch.tensor([tc],device='cuda'))[0,:8].float();t=(torch.arange(8,device='cuda',dtype=torch.float32)+.5)/8;e=2*torch.sinh(3*g)
  m3=(e+.95*t[:,None,None]).amax(0).half().float();m2=(e+.95*t[:,None,None]+(1-t[:,None,None])).amax(0).half().float()
  values=[h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(d)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx.square()+dy.square())/8),torch.asinh(m3/2)/3,torch.asinh(m2/2)/3]
  return torch.stack(values)[None].contiguous(memory_format=torch.channels_last)
 @torch.inference_mode()
 def predict(self,H,tx_row,tx_col):
  torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
  x,fs=self.features(H,tx_row,tx_col);im=self.frame(H,int(tx_row),int(tx_col))
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):st=self.net.encode(im)
  out=[]
  for lo in range(0,65536,self.chunk):
   v=x[None,lo:lo+self.chunk];pix=torch.arange(lo,min(lo+self.chunk,65536),device='cuda')[None]
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):delta=self.net.decode(st,v,pix).float()[0]
   out.append((self.net.last_base+delta)*20+fs[lo:lo+self.chunk])
  pred=torch.cat(out).cpu().numpy().T.reshape(2,256,256)
  assert np.isfinite(pred).all();return pred
