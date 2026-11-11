"""Input-only C47 inference. Raw H/TX -> teacher features -> learned residual."""
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from src.cc_core import gpu_guard
import numpy as np,torch
from scipy.ndimage import uniform_filter
from src.c34_model import Model as Teacher
from src.c47_model import Model
from src.c3_geometry import geometry
from src.c4_geometry import extra_geometry
from src.c8_tx_context import terrain_fields,tx_features
from src.c7_surface_features import terrain_fields as surface_fields,tx_visibility
from src.c9_extra_gpu import profile_gpu,surface_gpu
from src.c10_phase_geometry import phase_features_gpu

class Predictor:
    def __init__(self,checkpoint,teacher_directory=None):
        self.gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.6)
        checkpoint=Path(checkpoint);ck=torch.load(checkpoint,map_location='cuda',weights_only=False);cfg=ck['config']
        teacher_directory=Path(teacher_directory) if teacher_directory else R/'runs/c38_1024_continue'
        tc=torch.load(teacher_directory/'best.pt',map_location='cuda',weights_only=False);s=tc['model'];c=tc['config']
        self.teacher=Teacher(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval();self.teacher.load_state_dict(s)
        self.net=Model(cfg['kind'],s).cuda().eval();self.net.load_state_dict(ck['model'])
        yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
    @torch.inference_mode()
    def features(self,heightmap,tx_row,tx_col):
        h=np.array(heightmap,np.float32,copy=True);tr=int(tx_row);tc=int(tx_col)
        if h.shape!=(256,256) or not np.isfinite(h).all():raise ValueError('Expected finite256x256 H in metres')
        if tr!=tx_row or tc!=tx_col or not(0<=tr<256 and 0<=tc<256):raise ValueError('Expected integer TX pixel')
        tf=terrain_fields(h);sx=tf[1];sy=tf[2];hp=h-uniform_filter(h,size=5,mode='nearest')
        yy=self.yy;xx=self.xx;ht=float(h[tr,tc]);dx=xx-tc;dy=yy-tr;dz=h+1-ht-3
        dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
        basic=np.stack([h/100,(h-ht)/50,hp/5,sx/5,sy/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((yy==tr)&(xx==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)],0).reshape(11,-1).T
        hh=torch.from_numpy(h[None]).cuda();rr=torch.tensor([tr],device='cuda');cc=torch.tensor([tc],device='cuda')
        fg=torch.cat((geometry(hh,rr,cc),extra_geometry(hh,rr,cc)),1)[0].float().cpu().numpy();gg=fg.reshape(20,-1).T
        ctx=tx_features(tf,tr,tc)
        x=np.concatenate((basic,gg,np.broadcast_to(ctx,(65536,len(ctx)))),1)
        fs=(20*np.log10(4*np.pi*dist.reshape(-1,1)*np.array([415e6,5.775e9])[None]/299792458.)).astype(np.float32)
        crit=2.5+2*np.sinh(3*x[:,27]);phase=np.arange(8)/8;txh=x[:,31]
        extra=np.arcsinh((crit[:,None]-(3-np.remainder(txh[:,None]+phase,1)))/2)/3
        pv=profile_gpu(hh[0],tr,tc)
        sv=surface_gpu(hh[0],surface_fields(h),[tx_visibility(fg[:16],v) for v in [3,2]],tr,tc)
        x=np.concatenate((x,pv.cpu().numpy(),sv.cpu().numpy(),extra.astype(np.float32)),1)
        x=torch.from_numpy(x).cuda();x=torch.cat((x,phase_features_gpu(x)),1)
        return x,torch.from_numpy(fs).cuda()
    @torch.inference_mode()
    def predict_features(self,x,fs,precision='fp32'):
        # Frozen teacher exactly preserves the cache's4096-row BF16 convention.
        torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        base=[]
        for lo in range(0,len(x),4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):base.append(self.teacher(x[lo:lo+4096]).mean(1).float())
        base=torch.cat(base)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        delta=[]
        for lo in range(0,len(x),2048):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                delta.append(self.net(x[lo:lo+2048]).float())
        out=(base+torch.cat(delta))*20+fs
        assert torch.isfinite(out).all()
        return out
    @torch.inference_mode()
    def predict(self,heightmap,tx_row,tx_col):
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        x,fs=self.features(heightmap,tx_row,tx_col)
        out=self.predict_features(x,fs)
        return out.cpu().numpy().T.reshape(2,256,256)
