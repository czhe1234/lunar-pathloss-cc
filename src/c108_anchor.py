"""Score the identical24training-map anchor on every saved terrain-count checkpoint."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch
from src.c105_context import create,reader,ResidualPanel
from src.c108_train import evaluate
A=R/'audit/c108_diagnosis'
def main():
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 p=json.loads((A/'protocol.json').read_text());panel=ResidualPanel(reader(),p['sets']['8']['fit'],'context');out={};t=time.monotonic()
 for n in [8,32,128]:
  out[str(n)]={}
  for step in [200,500,1000,2000,4000]:
   ck=torch.load(R/f'runs/c108_n{n}/step{step}.pt',map_location='cpu',weights_only=False);assert ck['step']==step
   net=create().cuda().to(memory_format=torch.channels_last);net.load_state_dict(ck['state']);out[str(n)][str(step)]=evaluate(net,panel);del net;torch.cuda.empty_cache()
 atomic_json(A/'anchor_fit.json',dict(anchor_ids=panel.ids,metrics=out,GPU=gpu,seconds=time.monotonic()-t))
 print(json.dumps({n:{s:z['normalized_C82_SSE'] for s,z in v.items()} for n,v in out.items()}),flush=True)
if __name__=='__main__':main()
