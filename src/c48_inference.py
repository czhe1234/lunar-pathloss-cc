"""Single raw H/TX C48 interface, with no dataset/map-index or target access."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard
import numpy as np,torch
from src.c34_model import Model as Teacher
from src.c47_inference import Predictor as FeatureBuilder
from src.c48_model import Model
class Predictor(FeatureBuilder):
    def __init__(self,checkpoint,teacher_directory=None):
        self.gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.65)
        ck=torch.load(Path(checkpoint),map_location='cuda',weights_only=False);cfg=ck['config']
        root=Path(teacher_directory) if teacher_directory else R/'runs/c38_1024_continue'
        tc=torch.load(root/'best.pt',map_location='cuda',weights_only=False);s=tc['model'];c=tc['config']
        self.teacher=Teacher(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval()
        self.teacher.load_state_dict(s)
        fields=torch.zeros((1,256,256,3),device='cuda');info=torch.zeros((1,3),device='cuda',dtype=torch.long)
        self.net=Model(cfg['kind'],s,fields,info).cuda().eval();self.net.load_state_dict(ck['model'])
        yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
    @torch.inference_mode()
    def set_terrain(self,heightmap,tx_row,tx_col):
        h=torch.as_tensor(np.array(heightmap,np.float32,copy=True),device='cuda')
        sy,sx=torch.gradient(h,dim=(-2,-1))
        self.net.fields=torch.stack((h,sx,sy),-1)[None].contiguous()
        self.net.map_info=torch.tensor([[0,int(tx_row),int(tx_col)]],device='cuda',dtype=torch.long)
    @torch.inference_mode()
    def predict_features(self,x,fs,precision='fp32'):
        torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        base=[]
        for lo in range(0,len(x),4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):base.append(self.teacher(x[lo:lo+4096]).mean(1).float())
        base=torch.cat(base)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        correction=[]
        for lo in range(0,len(x),512):
            rows=x[lo:lo+512];maps=torch.zeros(len(rows),device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):correction.append(self.net(rows,maps).float())
        out=(base+torch.cat(correction))*20+fs
        assert torch.isfinite(out).all();return out
    @torch.inference_mode()
    def predict(self,heightmap,tx_row,tx_col):
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.set_terrain(heightmap,tx_row,tx_col)
        x,fs=self.features(heightmap,tx_row,tx_col)
        return self.predict_features(x,fs).cpu().numpy().T.reshape(2,256,256)
