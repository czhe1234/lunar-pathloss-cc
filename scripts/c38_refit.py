"""Fixed full official-TRAIN continuation from C37 full refit, selected only by C38 DEV.
No official validation targets or masks are loaded.
"""
import sys,json,math,time,argparse,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np
import torch
from src.c34_model import Model
p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--name',required=True);a=p.parse_args()
parent=R/'runs'/a.parent
score=json.loads((parent/'score_train_complete.json').read_text())
assert score['complete_requested_split'] and score['maps']==2562
assert all(x<y for x,y in zip(score['RMSE'],[2.680645907435645,3.031642312198183]))
cfg=json.loads((parent/'config.json').read_text());res=json.loads((parent/'result.json').read_text())
dense=cfg.get('dense',False)
if cfg.get('expanded',False):
 from src.c36_model import Model
gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.45);torch.set_num_threads(8)
torch.set_float32_matmul_precision('high');torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
o=R/'runs'/a.name;o.mkdir(exist_ok=False)
shutil.copy2(Path(__file__),o/'train_source.py');shutil.copy2(R/'src/c34_model.py',o/'model_source.py')
parts=[]
for part,folder in [('fit','c38_dense'),('dev','c38_dev_full')]:
 x=np.load(R/f'features/{folder}/{part}_x.npy',mmap_mode='r')
 y,fs,m=[np.load(R/f'features/{folder}/{part}_{v}.npy',mmap_mode='r') for v in ['y','fspl','mask']]
 parts.append((x,y,fs,m))
n=sum(len(v[0]) for v in parts)
X=torch.empty((n,294),dtype=torch.float32,device='cuda');Y=torch.empty((n,2),device='cuda');M=torch.empty_like(Y)
off=0
for x,y,fs,m in parts:
 for lo in range(0,len(x),131072):
  hi=min(lo+131072,len(x));sl=slice(off+lo,off+hi)
  X[sl]=torch.from_numpy(np.array(x[lo:hi])).cuda()
  Y[sl]=torch.from_numpy(np.array((y[lo:hi]-fs[lo:hi])/20)).cuda()
  M[sl]=torch.from_numpy(np.array(m[lo:hi],np.float32)).cuda()
 off+=len(x)
ck=torch.load(R/'runs/c37_full_refit/best.pt',map_location='cpu',weights_only=False);st=ck['model']
model=Model(cfg['mode'],cfg['members'],cfg['width'],st['embedding.mean'].cuda(),st['embedding.std'].cuda(),st['embedding.edges'].cuda(),st['bias'][0].cuda()*20).cuda()
model.load_state_dict(st)
opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=1e-4)
ratio=n/cfg['train_rows'];steps=round(res['best_step']*ratio);horizon=cfg['steps']*ratio;warmup=200*ratio
outcfg={**cfg,'name':a.name,'parent':a.parent,'full_refit':True,'full_parent':'c37_full_refit','memory_fraction_cap':.45,'train_rows':n,'original_FIT_rows':cfg['train_rows'],'selected_FIT_step':res['best_step'],'fixed_refit_steps':steps,'lr_horizon':horizon,'gpu':gpu,'official_VAL_used':False,'selection':'Same examples-per-map exposure as DEV-selected checkpoint, using all official TRAIN terrains; no early stopping'}
atomic_json(o/'config.json',outcfg)
gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);t0=time.time()
for step in range(1,steps+1):
 j=torch.randint(n,(cfg['batch'],),device='cuda',generator=gen)
 lr=min(1.,step/warmup)*(.00001+.5*(.0001-.00001)*(1+math.cos(math.pi*(step-1)/horizon)))
 for g in opt.param_groups:g['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):q=model(X[j])
 m=M[j,None];loss=((q-Y[j,None]).square()*m).sum((0,1))/(m.sum((0,1))*cfg['members']).clamp_min(1)
 loss.mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
 if step%1000==0 or step==steps:
  rec={'step':step,'steps':steps,'batch_RMSE_individual_members':(loss.sqrt()*20).detach().cpu().tolist(),'seconds':time.time()-t0,'lr':lr}
  atomic_json(o/'progress.json',rec);print(json.dumps(rec),flush=True)
torch.save({'model':model.state_dict(),'step':steps,'config':outcfg},o/'best.pt')
out={'status':'full_TRAIN_continuation_complete','maps':25720,'terrains':6120,'sampled_pixels':n,'steps':steps,'seconds':time.time()-t0,'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'official_validation_not_loaded':True}
atomic_json(o/'result.json',out);print(json.dumps(out),flush=True)
