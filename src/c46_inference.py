"""Fixed C38 plus learned residual field. Public input is H and TX only."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from cc_core import gpu_guard
from c34_inference import Predictor as PointPredictor
from c46_model import Model
import torch,numpy as np
from torch.nn import functional as F
class Predictor:
    def __init__(self,checkpoint,amp=False):
        self.gpu=gpu_guard();torch.set_num_threads(8);self.amp=amp
        self.point=PointPredictor(R/'runs/c38_1024_continue')
        torch.cuda.set_per_process_memory_fraction(.8)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.net=Model('path').cuda().to(memory_format=torch.channels_last);self.net.load_state_dict(ck['model']);self.net.eval()
        yy,xx=torch.meshgrid(torch.arange(256,device='cuda',dtype=torch.float32),torch.arange(256,device='cuda',dtype=torch.float32),indexing='ij')
        self.yy,self.xx=yy[None],xx[None]
    @torch.inference_mode()
    def predict_with_base(self,heightmap,tr,tc,base):
        h=torch.from_numpy(np.array(heightmap,np.float32,copy=True)[None]).cuda()
        ht=h[0,tr,tc];dx=self.xx-tc;dy=self.yy-tr;dz=h+1-ht-3
        dist=torch.sqrt(dx*dx+dy*dy+dz*dz).clamp_min(.25);sy,sx=torch.gradient(h,dim=(-2,-1))
        hp=h-F.avg_pool2d(F.pad(h[:,None],(2,2,2,2),mode='replicate'),5,stride=1)[:,0]
        x=torch.stack((h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,torch.log1p(dist)/6,((dx==0)&(dy==0)).float(),torch.exp(-(dx*dx+dy*dy)/8)),1)
        tx=torch.stack((ht*0+tc,ht*0+tr,ht+3))[None]
        freq=torch.tensor([415e6,5.775e9],device='cuda');fs=20*torch.log10(4*torch.pi*dist[:,None]*freq[None,:,None,None]/299792458.)
        base_tensor=torch.from_numpy(np.array(base,np.float32,copy=True)[None]).cuda()
        x=torch.cat((x,(base_tensor-fs)/20),1).contiguous(memory_format=torch.channels_last)
        torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):state=self.net.encode(x,tx)
        pred=[]
        for lo in range(0,65536,8192):
            pix=torch.arange(lo,min(lo+8192,65536),device='cuda')[None]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):q=self.net.decode(state,pix)
            pred.append(q.float()*20)
        delta=torch.cat(pred,1).transpose(1,2).reshape(1,2,256,256)
        return (base_tensor+delta)[0].cpu().numpy()
    @torch.inference_mode()
    def predict(self,heightmap,tx_row,tx_col):
        h=np.array(heightmap,np.float32,copy=True);tr=int(tx_row);tc=int(tx_col)
        assert h.shape==(256,256) and np.isfinite(h).all() and tr==tx_row and tc==tx_col and 0<=tr<256 and 0<=tc<256
        torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
        base=self.point.predict(h,tr,tc)
        result=self.predict_with_base(h,tr,tc,base);assert np.isfinite(result).all()
        return result
