"""Frozen H/TX interface for a terrain correction, a standalone graph field, or their fixed blend."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from src.cc_core import gpu_guard
from src.c3_geometry import basic_inputs
from src.c67_dense import model_inputs,ResidualMap
from src.c72_native import NativeMap
from src.c72_native_features import NativeFeatures
from src.c73_compact import compact
from src.c73_graph import GraphField
from src.c60_inference import Predictor as FullBaseline
import torch
import numpy as np


@torch.inference_mode()
def baseline_fields(base_model,h,ht,tr,tc):
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x,fs=base_model.features(h,tr,tc)
    g=x[:,11:31].T.reshape(1,20,256,256)
    # Reuse the exact 16+4 descriptors in the frozen baseline feature vector.
    sy,sx=torch.gradient(ht[0],dim=(-2,-1));base_model.high.fields.copy_(torch.stack((ht[0],sx,sy),-1)[None])
    base_model.high.map_info.copy_(torch.tensor([[0,tr,tc]],device='cuda',dtype=torch.long))
    t=(torch.arange(8,device='cuda',dtype=torch.float32)+.5)/8
    b=2*torch.sinh(3*g[0,:8].float());m1=(b+.95*t[:,None,None]).amax(0);m2=(b+.95*t[:,None,None]+(1-t[:,None,None])).amax(0)
    base_model.high.margins.copy_(torch.stack((m1,m2),-1).half()[None])
    base=base_model.predict_features(x,fs).T.reshape(1,2,256,256).contiguous()
    return base,g

class Predictor:
    def __init__(self,directory):
        self.gpu=gpu_guard();self.directory=Path(directory)
        self.cfg=json.loads((self.directory/'model.json').read_text())
        self.weight=float(self.cfg.get('graph_weight',0))
        assert 0<=self.weight<=1
        raw_gate=self.cfg.get('gate_weights')
        self.gate=torch.tensor(raw_gate,dtype=torch.float32,device='cuda') if raw_gate is not None else None
        if self.gate is not None:
            assert self.gate.shape==(2,4) and ((self.gate>=0)&(self.gate<=1)).all()
        self.need_base=self.weight<1 if self.gate is None else bool((self.gate<1).any())
        self.need_graph=self.weight>0 if self.gate is None else bool((self.gate>0).any())
        self.features=NativeFeatures()
        self.base=None;self.post=None;self.graph=None
        if self.need_base:
            self.base=FullBaseline(self.directory/self.cfg['baseline'])
            if self.cfg.get('fast_descriptors',False):
                from src.c74_features import FastDescriptors
                self.base.features=FastDescriptors()
            ck=torch.load(self.directory/self.cfg['correction'],map_location='cpu',weights_only=True)
            self.post=NativeMap(ck['channels'],ck['width'],ck['network']=='native').cuda().to(memory_format=torch.channels_last).eval()
            self.post.load_state_dict(ck['state'])
        if self.need_graph:
            ck=torch.load(self.directory/self.cfg['graph'],map_location='cpu',weights_only=True)
            self.graph=GraphField(56,ck['width'],ck['shuffle']).cuda().to(memory_format=torch.channels_last).eval()
            self.graph.load_state_dict(ck['state'])

    @torch.inference_mode()
    def predict(self,H,tr,tc):
        h=np.array(H,dtype=np.float32,copy=True)
        if h.shape!=(256,256) or not np.isfinite(h).all():raise ValueError('Expected a finite 256x256 terrain')
        if int(tr)!=tr or int(tc)!=tc or not(0<=tr<256 and 0<=tc<256):raise ValueError('Expected integer TX row and column, 0..255')
        tr,tc=int(tr),int(tc);ht=torch.as_tensor(h,device='cuda')[None];r=torch.tensor([tr],device='cuda');c=torch.tensor([tc],device='cuda')
        native=self.features(h,tr,tc).half()
        prediction=None
        v=None
        alpha=self.weight
        if self.need_graph or self.gate is not None:
            v=compact(native,self.features.geo.ids,self.features.geo.w).half().float()[None]
            if self.gate is not None:
                vis=v[:,16:17]
                zone=torch.where(vis>.99,0,torch.where(vis>=.01,1,torch.where(v[:,41:43]<-.8,3,2)))
                alpha=torch.stack([self.gate[b][zone[0,b]] for b in range(2)],0)[None]
        if self.base is not None:
            base,g=baseline_fields(self.base,h,ht,tr,tc)
            post_input=model_inputs(ht,r,c,base,g[:,:16],g[:,16:],native[None]).half().float().contiguous(memory_format=torch.channels_last)
            torch.set_float32_matmul_precision('high')
            with torch.autocast('cuda',torch.bfloat16):corr=self.post(post_input)
            prediction=(1-alpha)*(base+corr.float())
        if self.graph is not None:
            x,fs=basic_inputs(ht,r,c);prior=fs+(-20*(v[:,31:33]+v[:,41:43])).clamp(-8,48)
            torch.set_float32_matmul_precision('high')
            with torch.autocast('cuda',torch.bfloat16):
                output=prior+self.graph(torch.cat((x,v),1).contiguous(memory_format=torch.channels_last),ht,r,c).float()
            prediction=alpha*output if prediction is None else prediction+alpha*output
        if not torch.isfinite(prediction).all():raise RuntimeError('Nonfinite map')
        return prediction[0].cpu().numpy()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    model=Predictor(a.model);data=np.load(a.input,allow_pickle=False)
    np.save(a.output,model.predict(data['heightmap'],int(data['tx_row']),int(data['tx_col'])),allow_pickle=False)
