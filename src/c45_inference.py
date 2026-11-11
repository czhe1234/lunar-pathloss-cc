"""Input-only C43 interface: one raw heightmap and TX row/column, two PL maps."""
from pathlib import Path
import sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard
import numpy as np
import torch
from torch.nn import functional as F
from c45_model import Model
class Predictor:
    def __init__(self,checkpoint,amp=False):
        self.gpu=gpu_guard();torch.set_num_threads(8);self.amp=amp
        torch.set_float32_matmul_precision("high" if amp else "highest")
        torch.backends.cudnn.allow_tf32=amp;torch.backends.cuda.matmul.allow_tf32=amp
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.net=Model(ck['config']['variant']).cuda().to(memory_format=torch.channels_last)
        self.net.load_state_dict(ck['model']);self.net.eval()
        yy,xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij')
        self.yy,self.xx=yy[None],xx[None]
    @torch.inference_mode()
    def predict(self,heightmap,tx_row,tx_col):
        h=np.array(heightmap,np.float32,copy=True);tr=int(tx_row);tc=int(tx_col)
        assert h.shape==(256,256) and np.isfinite(h).all()
        assert tr==tx_row and tc==tx_col and 0<=tr<256 and 0<=tc<256
        h=torch.from_numpy(h[None]).cuda();ht=h[0,tr,tc]
        dx=self.xx-tc;dy=self.yy-tr;dz=h+1-ht-3;dist=torch.sqrt(dx*dx+dy*dy+dz*dz).clamp_min(.25)
        sy,sx=torch.gradient(h,dim=(-2,-1))
        hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0]
        x=torch.stack((h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(dist)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx*dx+dy*dy)/8)),1).contiguous(memory_format=torch.channels_last)
        tx=torch.stack((ht*0+tc,ht*0+tr,ht+3))[None]
        freq=torch.tensor([415e6,5.775e9],device='cuda')
        fs=20*torch.log10(4*torch.pi*dist[:,None]*freq[None,:,None,None]/299792458.)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):state=self.net.encode(x,tx)
        ps=[]
        for lo in range(0,65536,8192):
            pix=torch.arange(lo,min(lo+8192,65536),device='cuda')[None]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):q=self.net.decode(state,pix)
            ps.append(q.float()*20)
        pred=torch.cat(ps,1).transpose(1,2).reshape(1,2,256,256)+fs
        result=pred[0].cpu().numpy()
        assert np.isfinite(result).all()
        return result
