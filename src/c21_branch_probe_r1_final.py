"""Frozen inference perturbations on four DEV maps; CPU only; no optimization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import numpy as np
import torch
from c21_model_r1 import Model
from c21_data import Data,R
torch.set_num_threads(4);run=R/'runs/c21r1_pair_s20261123';cfg=json.loads((run/'config.json').read_text());norm=np.load(R/'audit/c21_execution_20261101/normalization.npz');d=Data();ids=cfg['dev_panel'][:4];samples=[d.sample(i) for i in ids];pix=torch.from_numpy(np.random.default_rng(20261139).choice(65536,size=1024,replace=False).astype(np.int64))[None]
results={};t0=time.time()
with torch.inference_mode():
 for label,geo in [('A',False),('B',True)]:
  net=Model(torch.from_numpy(norm['mean']),torch.from_numpy(norm['std']),geometry=geo);ck=torch.load(run/f'best_{label}.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);step=ck['step'];del ck;net.eval();states=[]
  for x,tx,fs,y,m in samples:states.append(net.encode(torch.from_numpy(x)[None],torch.from_numpy(tx)[None]))
  preds={v:[] for v in ['full','zero_global_values','foreign_global_tokens','zero_geometry_bias']};yy=[];mm=[];token_std=[]
  for j,(x,tx,fs,y,m) in enumerate(samples):
   st=states[j];base=torch.from_numpy(fs).flatten(1).T[pix[0]][None]
   yy.append(y.reshape(2,-1)[:,pix[0].numpy()].T);mm.append(m.reshape(2,-1)[:,pix[0].numpy()].T)
   preds['full'].append((net.decode(st,pix,256)*20+base).numpy()[0])
   kv=st[3].clone();kv[1].zero_();modified=(*st[:3],kv,*st[4:]);preds['zero_global_values'].append((net.decode(modified,pix,256)*20+base).numpy()[0])
   modified=(*st[:3],states[(j+1)%len(states)][3],*st[4:]);preds['foreign_global_tokens'].append((net.decode(modified,pix,256)*20+base).numpy()[0])
   w=net.geoweight.clone();net.geoweight.zero_();preds['zero_geometry_bias'].append((net.decode(st,pix,256)*20+base).numpy()[0]);net.geoweight.copy_(w)
   token_std.append(float(st[3][1].std(dim=2).mean()))
  yy=np.array(yy,dtype=np.float64);mm=np.array(mm,dtype=np.float64);pp={k:np.array(v,dtype=np.float64) for k,v in preds.items()};result={'checkpoint_step':step,'geometry_weight_norm':float(net.geoweight.norm()),'valid_count':mm.sum((0,1)).tolist(),'value_token_spatial_std':token_std,'variants':{}}
  for name,p in pp.items():
   result['variants'][name]={'rmse':np.sqrt((((p-yy)**2)*mm).sum((0,1))/mm.sum((0,1))).tolist(),'prediction_shift_rms_db':np.sqrt((((p-pp['full'])**2)*mm).sum((0,1))/mm.sum((0,1))).tolist()}
  results[label]=result;print(json.dumps({label:result}),flush=True)
out={'scope':'four fixed DEV maps,1024 uniform pixels each; frozen weights; CPU FP32; inference perturbations, not retrained ablations','ids':ids,'seconds':time.time()-t0,'models':results}
(R/'audit/c21_execution_20261101/r1_branch_probe_final.json').write_text(json.dumps(out,indent=2)+'\n')
