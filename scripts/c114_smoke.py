import sys,json,time
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch
from src.c105_context import create,reader
from src.c114_inputs import InputPanel
g=gpu_guard();torch.set_num_threads(4)
p=json.loads((R/'audit/c114_input_convention/protocol.json').read_text());d=InputPanel(reader(),p['sets']['128']['fit'][:8],'context')
InputPanel.geometry_mode='old';a,b,y,w=d.batch(slice(0,8))
InputPanel.geometry_mode='consistent';c,bb,yy,ww=d.batch(slice(0,8))
assert torch.equal(b,bb) and torch.equal(y,yy) and torch.equal(w,ww)
changed=(a!=c).flatten(2).any(2).any(0).nonzero().flatten().tolist();assert set(changed)<=set([1,7,8,56,57])
n=create().cuda().to(memory_format=torch.channels_last);o=torch.optim.AdamW(n.parameters(),lr=.0004,weight_decay=.01,fused=True)
with torch.autocast('cuda',torch.bfloat16):assert torch.count_nonzero(n(a))==0 and torch.count_nonzero(n(c))==0
times=[]
for j in range(8):
 t=time.monotonic()
 with torch.autocast('cuda',torch.bfloat16):pred=b+n(c).float()
 e=torch.where(w,pred-y,0.);loss=(e.square().sum((0,2,3))/w.sum((0,2,3))).mean()/25
 o.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss);o.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
z=dict(changed_channels=changed,GPU=g,seconds_per_update=sum(times[3:])/len(times[3:]),finite_loss=float(loss))
atomic_json(R/'audit/c114_input_convention/smoke.json',z);print(z,flush=True)
