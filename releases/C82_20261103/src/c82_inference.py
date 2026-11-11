"""Frozen C82 predictor. It reuses the exact C76 forward and reads each component once."""
from pathlib import Path
import json,sys
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c74_inference import Predictor as Parent
from src.c82_blend import field_features

class Predictor(Parent):
    def __init__(self,directory):
        super().__init__(directory)
        assert self.weight==.5 and self.gate is None
        cfg=json.loads((self.directory/'spectrum.json').read_text())
        self.mode=cfg['mode'];self.beta=[torch.tensor(v,device='cuda',dtype=torch.float32) for v in cfg['beta']]
        self.xy=torch.meshgrid(torch.arange(256,device='cuda'),torch.arange(256,device='cuda'),indexing='ij')
        self.freq=torch.tensor([415e6,5.775e9],device='cuda')[None,:,None,None]*(4*np.pi/299792458)
        core=self.base.predict_features
        def retain_base(*args,**kwargs):
            v=core(*args,**kwargs)
            self.base_field=v.T.reshape(1,2,256,256).contiguous()
            return v
        self.base.predict_features=retain_base
        def retain_local(module,inputs,v):
            self.local_field=self.base_field+v.float()
        def retain_graph(module,args,v):
            compact=args[0][:,11:]
            self.vis=compact[:,16:17].float()
        self.post.register_forward_hook(retain_local)
        self.graph.register_forward_hook(retain_graph)
    @torch.inference_mode()
    def predict(self,height,tr,tc):
        raw=super().predict(height,tr,tc)
        p=torch.as_tensor(raw,device='cuda')[None]
        graph=2*p-self.local_field
        h=torch.as_tensor(np.asarray(height,'float32'),device='cuda')
        yy,xx=self.xy
        altitude=h[min(int(tr)+1,255),int(tc)].floor()+3
        d=torch.sqrt((xx-int(tc)).square()+(yy-int(tr)).square()+(h+1-altitude).square()).clamp_min(.25)
        fs=20*torch.log10(d[None,None]*self.freq)
        corr=torch.cat([(field_features(self.base_field,self.local_field,graph,fs,self.vis,self.mode,b)*
                         self.beta[b][None,:,None,None]).sum(1,keepdim=True) for b in (0,1)],1).clamp(-15,15)
        v=p+corr
        if not torch.isfinite(v).all():raise RuntimeError('Nonfinite C82 output')
        return v[0].cpu().numpy()
