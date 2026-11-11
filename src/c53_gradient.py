"""FIT-only shared-backbone gradient conflict across the two frequency tasks."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c47_train import load_teacher
from src.c34_model import Model
import numpy as np,torch
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.25);torch.set_float32_matmul_precision('high')
s,c=load_teacher();net=Model(c['mode'],c['members'],c['width'],s['embedding.mean'],s['embedding.std'],s['embedding.edges'],s['bias'][0]*20).cuda().eval();net.load_state_dict(s)
X=np.load(R/'features/c38_dense/fit_x.npy',mmap_mode='r');Y=np.load(R/'features/c38_dense/fit_y.npy',mmap_mode='r');F=np.load(R/'features/c38_dense/fit_fspl.npy',mmap_mode='r');M=np.load(R/'features/c38_dense/fit_mask.npy',mmap_mode='r')
ids=np.random.default_rng(20261102).choice(len(X),131072,replace=False)
x=torch.from_numpy(np.array(X[ids])).cuda();y=torch.from_numpy((np.array(Y[ids])-np.array(F[ids]))/20).cuda();m=torch.from_numpy(np.array(M[ids],np.float32)).cuda()
params=[(n,p) for n,p in net.named_parameters() if n.startswith(('embedding.','layers.'))]
avg=[torch.zeros(sum(p.numel() for _,p in params),device='cuda') for _ in range(2)];cos=[]
for lo in range(0,len(ids),4096):
 with torch.autocast('cuda',dtype=torch.bfloat16):q=net(x[lo:lo+4096])
 mm=m[lo:lo+4096,None];loss=((q-y[lo:lo+4096,None]).square()*mm).sum((0,1))/(mm.sum((0,1))*net.k).clamp_min(1)
 grad=[]
 for b in range(2):
  gs=torch.autograd.grad(loss[b],[p for _,p in params],retain_graph=b==0)
  v=torch.cat([g.float().reshape(-1) for g in gs]);grad.append(v);avg[b]+=v/32
 cos.append(float(torch.nn.functional.cosine_similarity(grad[0],grad[1],dim=0)))
cosine=float(torch.nn.functional.cosine_similarity(avg[0],avg[1],dim=0))
out={'gpu':gpu,'FIT_points':len(ids),'shared_gradients':True,'dropout_during_measurement':False,'batch_gradient_cosine_mean':float(np.mean(cos)),'mean_gradient_cosine':cosine,'norms':[float(a.norm()) for a in avg],
 'interpretation':'Negative mean-gradient cosine indicates antagonistic local updates; does not by itself prove frequency separation improves DEV.',
 'run_separate_band_screen_if_cosine_below_minus020':cosine<-.20,'official_VAL_used':False}
A=R/'audit/c53_frequency_specialization_20261102';A.mkdir(exist_ok=True);atomic_json(A/'gradient.json',out);print(json.dumps(out),flush=True)
