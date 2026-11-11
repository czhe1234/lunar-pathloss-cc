"""Frozen-node-model input substitution; no optimizer and no new selection."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c130_probe_model import Panel,create,evaluate,A
import torch
g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');start=time.monotonic()
train=Panel('fit');mon=Panel('monitor');net=create(train);out=A/'ablation';out.mkdir(exist_ok=False);results={}
for checkpoint in ['best','last']:
 ck=torch.load(R/f'runs/c130_nodes_nodes/{checkpoint}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
 results[checkpoint]=dict(step=ck['step'],cases={})
 for mode in ['nodes','old','shuffled']:
  z={}
  for name,panel in [('fit',train),('monitor',mon)]:
   z[name]=evaluate(net,panel,mode,save=out/f'{checkpoint}_{mode}_{name}')
  results[checkpoint]['cases'][mode]=z;print(checkpoint,mode,z['monitor']['tail_RMSE'],flush=True)
atomic_json(A/'branch_audit.json',dict(status='complete',GPU=g,results=results,seconds=time.monotonic()-start,
 meaning='fixedweightsinputsubstitutiondiagnostic; zero/shuffled can be out-of-distribution;not proof of no raw information or a deployable selection rule',official_val_used=False))
