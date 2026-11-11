"""Fixed-band composite. Only a raw heightmap and integer TX pixel are inputs."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard
import numpy as np,torch
from src.c34_model import Model as Teacher
from src.c47_model import Model as Low
from src.c49_model import Model as High49
from src.c52_model import Model as High52
from src.c47_inference import Predictor as Features
from src.c3_geometry import geometry
class Predictor(Features):
    def __init__(self,model_directory):
        self.gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.65)
        d=Path(model_directory);self.manifest=json.loads((d/'manifest.json').read_text())
        tc=torch.load(d/'teacher.pt',map_location='cpu',weights_only=False);s=tc['model'];c=tc['config']
        self.teacher=Teacher(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval();self.teacher.load_state_dict(s)
        lc=torch.load(d/'low.pt',map_location='cpu',weights_only=False);hc=torch.load(d/'high.pt',map_location='cpu',weights_only=False)
        self.low=Low('attention',s).cuda().eval();self.low.load_state_dict(lc['model'])
        cls=High52 if self.manifest['high_architecture']=='c52' else High49
        fields=torch.zeros((1,256,256,3),device='cuda');info=torch.zeros((1,3),device='cuda',dtype=torch.long);margins=torch.zeros((1,256,256,2),device='cuda',dtype=torch.float16)
        self.high=cls('transport',s,fields,info,margins).cuda().eval();self.high.load_state_dict(hc['model'])
        yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
    @torch.inference_mode()
    def set_terrain(self,H,tr,tc):
        h=torch.from_numpy(np.array(H,np.float32,copy=True)).cuda();sy,sx=torch.gradient(h,dim=(-2,-1))
        self.high.fields=torch.stack((h,sx,sy),-1)[None].contiguous()
        self.high.map_info=torch.tensor([[0,int(tr),int(tc)]],device='cuda',dtype=torch.long)
        g=geometry(h[None],torch.tensor([tr],device='cuda'),torch.tensor([tc],device='cuda'))[0,:8].float()
        t=(torch.arange(8,device='cuda',dtype=torch.float32)+.5)/8;b=2*torch.sinh(3*g)
        self.high.margins=torch.stack(((b+.95*t[:,None,None]).amax(0),(b+.95*t[:,None,None]+(1-t[:,None,None])).amax(0)),-1).half()[None]
    @torch.inference_mode()
    def predict_features(self,x,fs,precision='fp32'):
        torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        base=[]
        for lo in range(0,len(x),4096):
            with torch.autocast('cuda',dtype=torch.bfloat16):base.append(self.teacher(x[lo:lo+4096]).mean(1).float())
        base=torch.cat(base)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        corrections=[]
        for lo in range(0,len(x),512):
            v=x[lo:lo+512];maps=torch.zeros(len(v),device='cuda',dtype=torch.long)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                qlo=self.low(v)[:,0].float();qhi=self.high(v,maps)[:,1].float()
            corrections.append(torch.stack((qlo,qhi),-1))
        out=(base+torch.cat(corrections))*20+fs
        assert torch.isfinite(out).all();return out
    @torch.inference_mode()
    def predict(self,H,tx_row,tx_col):
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.set_terrain(H,tx_row,tx_col);x,fs=self.features(H,tx_row,tx_col)
        return self.predict_features(x,fs).cpu().numpy().T.reshape(2,256,256)
