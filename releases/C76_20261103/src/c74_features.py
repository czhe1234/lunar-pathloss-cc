"""Build exactly the existing 294 descriptors, keeping dense concatenations on the selected GPU."""
import ctypes
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from src.c3_geometry import geometry
from src.c4_geometry import extra_geometry
from src.c8_tx_context import terrain_fields,tx_features
from src.c7_surface_features import RADII,AREA,tx_visibility
from src.c9_extra_gpu import profile_gpu
from src.c10_phase_geometry import phase_features_gpu
class FastDescriptors:
    def __init__(self):
        angles=2*np.pi*np.arange(16)/16;parts=[]
        for a in angles:
            parts.append(np.stack((np.cos(a)*RADII,np.sin(a)*RADII,RADII,AREA,np.full(16,np.cos(a)),np.full(16,np.sin(a))),1))
        self.sampling=torch.as_tensor(np.ascontiguousarray(parts),device='cuda',dtype=torch.float64)
        self.lib=ctypes.CDLL(str(Path(__file__).with_name('libc74_surface.so')))
        self.fn=self.lib.surface74;p=ctypes.c_void_p
        self.fn.argtypes=[p,p,ctypes.c_int,ctypes.c_int,p,p];self.fn.restype=ctypes.c_int
        yy,xx=np.mgrid[:256,:256];self.yy=yy.astype(np.float32);self.xx=xx.astype(np.float32)
    @torch.inference_mode()
    def __call__(self,H,tr,tc):
        h=np.array(H,np.float32,copy=True);tf=terrain_fields(h)
        ht=float(h[tr,tc]);dx=self.xx-tc;dy=self.yy-tr;dz=h+1-ht-3;dist=np.sqrt(dx*dx+dy*dy+dz*dz).clip(.25)
        hp=h-uniform_filter(h,size=5,mode='nearest')
        basic=np.stack((h/100,(h-ht)/50,hp/5,tf[1]/5,tf[2]/5,dx/128,dy/128,dz/50,np.log1p(dist)/6,((self.yy==tr)&(self.xx==tc)).astype(np.float32),np.exp(-(dx*dx+dy*dy)/8)),0)
        fs=(20*np.log10(4*np.pi*dist.reshape(-1,1)*np.array([415e6,5.775e9])[None]/299792458.)).astype(np.float32)
        hgpu=torch.from_numpy(h).cuda();r=torch.tensor([tr],device='cuda');c=torch.tensor([tc],device='cuda')
        geo=torch.cat((geometry(hgpu[None],r,c),extra_geometry(hgpu[None],r,c)),1)[0].float()
        arr=geo.cpu().numpy()
        vis=[tx_visibility(arr[:16],th) for th in [3,2]]
        fields=torch.from_numpy(np.concatenate((np.stack((h,tf[1],tf[2])),*vis),axis=0).astype(np.float32,copy=False)).cuda()
        surf=torch.empty((65536,40),device='cuda')
        rc=self.fn(fields.data_ptr(),self.sampling.data_ptr(),tr,tc,surf.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError(f'Fused surface feature error {rc}')
        ctx=tx_features(tf,tr,tc);txh=ctx[0]
        critical=(2.5+2*np.sinh(3*arr[16])).reshape(-1,1);phase=np.arange(8)/8
        extra=np.arcsinh((critical-(3-np.remainder(txh+phase,1)))/2)/3
        to_gpu=lambda x:torch.from_numpy(np.ascontiguousarray(x)).cuda()
        profile=profile_gpu(hgpu,tr,tc)
        vals=torch.cat((to_gpu(basic).reshape(11,-1).T,geo.reshape(20,-1).T,to_gpu(ctx)[None].expand(65536,-1),
             profile,surf,to_gpu(extra.astype(np.float32))),1)
        return torch.cat((vals,phase_features_gpu(vals)),1),to_gpu(fs)
