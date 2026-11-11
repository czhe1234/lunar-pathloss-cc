import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import numpy as np,torch
from c21_data import Data,R
from c21_model import Model
torch.set_num_threads(4);run=R/'runs/c21_pair_s20261123';cfg=json.loads((run/'config.json').read_text());norm=np.load(R/'audit/c21_execution_20261101/normalization.npz');d=Data();i=cfg['dev_panel'][0];x,tx,fs=d.inputs(i);results={}
def stat(o,axis):
 s=o.float().std(dim=axis);mean=o.float().mean(dim=axis)
 return {'spatial_std_mean':float(s.mean()),'spatial_std_rms':float(s.square().mean().sqrt()),'mean_magnitude_rms':float(mean.square().mean().sqrt()),'global_rms':float(o.float().square().mean().sqrt())}
with torch.inference_mode():
 for stage in ['initial','step3000']:
  torch.manual_seed(cfg['seed']);net=Model(torch.from_numpy(norm['mean']),torch.from_numpy(norm['std']),geometry=True)
  if stage!='initial':
   ck=torch.load(run/'best_B.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);step=ck['step'];del ck
  else:step=0
  net.eval();obs={'step':step};handles=[]
  for j,mod in enumerate(net.encoder):handles.append(mod.register_forward_hook(lambda m,a,o,j=j:obs.update({f'encoder{j}':stat(o.flatten(2),2)})))
  for j,mod in enumerate(net.scene):handles.append(mod.register_forward_hook(lambda m,a,o,j=j:obs.update({f'scene{j}':stat(o,1)})))
  handles.append(net.knorm.register_forward_hook(lambda m,a,o:obs.update({'knorm':stat(o,1)})))
  st=net.encode(torch.from_numpy(x)[None],torch.from_numpy(tx)[None]);obs['keys']=stat(st[3][0],2);obs['values']=stat(st[3][1],2)
  results[stage]=obs
  for h in handles:h.remove()
(R/'audit/c21_execution_20261101/token_probe.json').write_text(json.dumps({'map':i,'scope':'one fixed DEV map; CPU FP32; random initialization vs frozen checkpoint','results':results},indent=2)+'\n')
print(json.dumps(results),flush=True)
