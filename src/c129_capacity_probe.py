"""Fixed checkpoint comparison on shared16anchors and original128monitor."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c127_train import reader,Panel,evaluate
from src.c129_capacity_model import initialize
A=R/'audit/c129_fitgap'
g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');start=time.monotonic()
p=json.loads((A/'capacity_protocol.json').read_text());r=reader();anchors=Panel(r,p['fit16'],reference=True);monitor=Panel(r,p['monitor'],reference=True)
cases=[('n16',250),('n16',1000),('n64',250),('w64',250)];results={}
for name,ep in cases:
 path=R/f'runs/c129_capacity_{name}/ep{ep}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
 width=p['arms'][name]['width'];net=initialize(width);net.load_state_dict(ck['state']);net=net.cuda().to(memory_format=torch.channels_last)
 fit=evaluate(net,anchors,'residual',16);mon=evaluate(net,monitor,'residual',16)
 key=f'{name}_ep{ep}';results[key]=dict(exposures=ep,updates=ck['updates'],shared16anchor=fit,monitor=mon,gain_vs_C82=(np.array(mon['C82'])-mon['rmse']).tolist())
 print('PROBE',key,fit['rmse'],mon['rmse'],flush=True);del net;torch.cuda.empty_cache()
out=dict(status='complete',GPU=g,scope='fixed250exposures and1000updates; same16FITanchors andoriginal128monitor; no officialVAL,no training decisions usingmonitor duringcapacityrun',cases=results,seconds=time.monotonic()-start)
atomic_json(A/'capacity_probe.json',out);print('DONE',json.dumps(out),flush=True)
