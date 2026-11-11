import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import gpu_guard,ROOT,atomic_json,d4,d4_coords
from c3_geometry import geometry,make_inputs
from c3_model import Model
import torch
info=gpu_guard();torch.set_num_threads(8);torch.manual_seed(31)
h=torch.zeros(1,256,256,device='cuda');r=torch.tensor([128],device='cuda');c=torch.tensor([128],device='cuda')
t0=time.time();g=geometry(h,r,c);torch.cuda.synchronize();sec=time.time()-t0
assert torch.isfinite(g).all();assert (g[:,:8]<0).all();assert (g[:,9]==0).all();assert (g[:,12:15]>.90).all()
# translation of absolute elevation leaves geometric descriptors and FSPL unchanged.
g2=geometry(h+37,r,c);assert torch.allclose(g,g2,atol=.002)
x,fs=make_inputs(h,r,c,g);x2,fs2=make_inputs(h+37,r,c,g2);assert torch.allclose(fs,fs2,atol=1e-5)
# A ridge blocks a ray beyond it but leaves the near side clear.
h[:,:,150:155]=20;gr=geometry(h,r,c);e=2*torch.sinh(gr[:,:8].float()*3)
assert e[0,:,128,200].max()>5;assert e[0,:,128,140].max()<0
# D4 reversal of scalar features; signed normal incidence is invariant.
hr=d4(h,1);rr,cc=d4_coords(r,c,1);gg=geometry(hr,rr,cc)
diff=(gg-d4(gr,1)).abs();assert diff[:,:15].max()<.015, float(diff[:,:15].max())
model=Model().cuda().to(memory_format=torch.channels_last)
with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x)
assert p.shape==(1,2,256,256) and torch.equal(p,torch.zeros_like(p))
loss=(p+fs-(fs+5)).square().mean();loss.backward();assert model.head[-1].weight.grad.abs().sum()>0
result={'passed':True,'device':info,'flat_ridge_translation_d4_and_backward':True,'D4_max_diff':float(diff.max()),'geometry_single_cold_seconds':sec,'parameters':sum(a.numel() for a in model.parameters())}
atomic_json(ROOT/'audit/c3_checks.json',result);print(json.dumps(result))
