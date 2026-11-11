"""Same-receiver/source geometry, coarse4m versus native1m source resolution."""
import sys,json,hashlib,argparse,math,time,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c99_axial import AxialField
from src.c101_train import create
from src.c102_transport import Transport
from src.c102_data import Bank
A=R/'audit/c102_transport'
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load_encoders():
 point_ck=torch.load(R/'runs/c101_coarse/best.pt',map_location='cpu',weights_only=False);nm=np.load(R/'audit/c101_residual/normalization.npz')
 point=create(torch.tensor(nm['mean']),torch.tensor(nm['std'])).cuda();point.load_state_dict(point_ck['state']);point.eval();point.requires_grad_(False)
 source=AxialField(True).cuda().to(memory_format=torch.channels_last);source.load_state_dict(torch.load(R/'runs/c99_global/best.pt',map_location='cpu',weights_only=False)['state']);source.eval();source.requires_grad_(False)
 return point,source
@torch.inference_mode()
def evaluate(net,bank,point,scale):
 sums=torch.zeros(6,device='cuda',dtype=torch.float64)
 for start in range(0,len(bank.ids),8):
  maps=torch.arange(start,min(len(bank.ids),start+8),device='cuda')
  for sub in range(0,256,128):
   pos=bank.monitor_positions[maps,sub:sub+128]
   q,n,bias,mass,base,t,w=bank.batch(maps,pos,scale,point,net is not None)
   if net is None:p=base
   else:
    with torch.autocast('cuda',torch.bfloat16):p=base+net(q,n,bias,mass)
   mask=t[...,2:4].bool();err=torch.where(mask,p-t[...,:2],0.)
   indices=maps[:,None]*bank.pp+pos;c82=bank.points.base[indices];be=torch.where(mask,c82-t[...,:2],0.)
   sums+=torch.cat([err.square().sum((0,1),dtype=torch.float64),be.square().sum((0,1),dtype=torch.float64),mask.sum((0,1))])
 r=sums.cpu().numpy();return dict(rmse=np.sqrt(r[:2]/r[4:]).tolist(),C82=np.sqrt(r[2:4]/r[4:]).tolist(),sse=r[:2].tolist(),count=r[4:].tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['coarse','native'],required=True);ap.add_argument('--steps',type=int,default=4000);a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 plan=json.loads((A/'protocol.json').read_text());assert a.steps==plan['steps']
 point,source=load_encoders();torch.manual_seed(102041);net=Transport();h=digest(net);assert h==plan['initial_state_sha256'];net=net.cuda()
 out=R/'runs'/('c102_'+a.mode);out.mkdir(exist_ok=False);t=time.monotonic();scale=4 if a.mode=='coarse' else 1
 train=Bank('fit',source);mon=Bank('monitor',source);del source;torch.cuda.empty_cache()
 cfg=dict(**vars(a),GPU=gpu,parameters=sum(p.numel() for p in net.parameters()),initial_state_sha256=h,source_scale=scale,
  train_map_batch=8,query_per_map=128,monitor_points_per_map=256,frozen_point='runs/c101_coarse/best.pt',frozen_source='runs/c99_global/best.pt',
  input='H/TX-derived fields and frozenC82, no masks/labels in inputs',official_val_used=False)
 atomic_json(out/'config.json',cfg)
 initial=evaluate(None,mon,point,scale);atomic_json(out/'initial.json',initial);best=float(np.mean(initial['rmse']));best_step=0
 torch.save(dict(state=net.state_dict(),step=0,config=cfg),out/'best.pt')
 print('INITIAL',a.mode,initial,'loadseconds',time.monotonic()-t,flush=True)
 opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.005,fused=True)
 gen=torch.Generator(device='cuda').manual_seed(102073);history=[]
 for step in range(1,a.steps+1):
  maps=torch.randint(len(train.ids),(8,),device='cuda',generator=gen);pos=torch.randint(train.pp,(8,128),device='cuda',generator=gen)
  q,n,bias,mass,base,target,weight=train.batch(maps,pos,scale,point)
  warm=min(1.,step/100);lr=.0008*warm*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
  for pg in opt.param_groups:pg['lr']=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',torch.bfloat16):p=base+net(q,n,bias,mass)
  mask=target[...,2:4];wm=mask*weight[...,None];err=(p-target[...,:2]).square()
  loss=((err*wm).sum((0,1))/wm.sum((0,1)).clamp_min(1)).mean()/25
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step();assert torch.isfinite(loss)
  if step%100==0:
   atomic_json(out/'progress.json',dict(step=step,total_steps=a.steps,loss=float(loss),seconds=time.monotonic()-t))
  if step%500==0 or step==a.steps:
   net.eval();val=evaluate(net,mon,point,scale);net.train();criterion=float(np.mean(val['rmse']))
   if criterion<best:
    best=criterion;best_step=step;torch.save(dict(state=net.state_dict(),step=step,config=cfg),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   row=dict(step=step,monitor=val,gain_vs_C82=(np.array(val['C82'])-val['rmse']).tolist(),best_step=best_step,seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
   history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',row);print(a.mode,json.dumps(row),flush=True)
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);net.eval()
 val=evaluate(None if best_step==0 else net,mon,point,scale)
 atomic_json(out/'result.json',dict(status='complete',best_step=best_step,last_step=a.steps,monitor=val,gain_vs_C82=(np.array(val['C82'])-val['rmse']).tolist(),seconds=time.monotonic()-t,GPU=gpu,official_val_used=False))
 print('DONE',a.mode,best_step,val,flush=True)
if __name__=='__main__':main()
