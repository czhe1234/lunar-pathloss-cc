"""Direct frozen C82 residuals with physical grazing-incidence numerical embeddings."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c34_model import Model
A=R/'audit/c103_terrain_structure'
def digest(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def create(mean,std):
 torch.manual_seed(103041)
 net=Model('periodic',8,512,mean,std,torch.zeros(235,17),torch.zeros(2))
 with torch.no_grad():
  net.embedding.frequencies.copy_(torch.tensor([.0625,.125,.25,.5,1.,2.,4.,8.])[None].expand(235,-1))
  net.head.zero_();net.bias.zero_()
 return net
from src.c101_train import Data as PointData
class Data(PointData):
 def __init__(self,part,mode):
  super().__init__(part,'coarse');self.mode=mode
  import csv
  rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
  theta=np.load(A/'theta.npy')
  self.theta=torch.from_numpy(np.array(theta[[int(rows[i]['hm_row']) for i in self.meta['ids']]],np.float32)).cuda()
 def x(self,ids):
  if isinstance(ids,slice):ix=torch.arange(ids.start or 0,min(ids.stop or len(self.common),len(self.common)),device='cuda')
  else:ix=ids
  common=torch.cat([self.common[ix],self.extra[ix]],1)
  values=self.theta[ix//self.meta['points_per_map']] if self.mode=='terrain' else torch.zeros((len(ix),66),device='cuda')
  return torch.cat([common,values],1)
@torch.inference_mode()
def score(net,d):
 net.eval();s=torch.zeros(2,device='cuda',dtype=torch.float64);c=s.clone()
 for start in range(0,len(d.common),32768):
  ix=slice(start,start+32768);x=d.x(ix)
  with torch.autocast('cuda',torch.bfloat16):p=d.base[ix]+20*net(x).mean(1)
  mask=d.target[ix,2:4].bool();e=torch.where(mask,p-d.target[ix,:2],0.)
  s+=e.square().sum(0,dtype=torch.float64);c+=mask.sum(0)
 net.train();return dict(rmse=(s/c).sqrt().cpu().tolist(),sse=s.cpu().tolist(),count=c.cpu().tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['local','terrain'],default='terrain');ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--batch',type=int,default=16384);a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 if a.smoke:
  rec=[]
  for batch in [4096,8192,16384]:
   net=create(torch.zeros(235),torch.ones(235)).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001,fused=True)
   x=torch.randn(batch,235,device='cuda');y=torch.randn(batch,1,2,device='cuda')*.1
   torch.cuda.reset_peak_memory_stats();tt=[]
   for j in range(12):
    torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',torch.bfloat16):pred=net(x)
    loss=(pred-y).square().mean();loss.backward();assert torch.isfinite(loss);opt.step();torch.cuda.synchronize();tt.append(time.monotonic()-t)
   row=dict(batch=batch,seconds_per_step=float(np.mean(tt[3:])),samples_per_second=batch/float(np.mean(tt[3:])),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
   print(row,flush=True);rec.append(row);del net,opt,x,y,pred,loss;torch.cuda.empty_cache()
  A.mkdir(exist_ok=True);atomic_json(A/'smoke.json',dict(GPU=gpu,measurements=rec));return
 plan=json.loads((A/'protocol.json').read_text());assert a.steps==plan['steps'] and a.batch==plan['batch']
 nm=np.load(A/'normalization.npz');net=create(torch.tensor(nm['mean']),torch.tensor(nm['std']))
 init_hash=digest(net);assert init_hash==plan['initial_state_sha256']
 out=R/'runs'/('c103_'+a.mode);out.mkdir(exist_ok=False)
 net=net.cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.001,fused=True)
 t=time.monotonic();train=Data('fit',a.mode);monitor=Data('monitor',a.mode)
 cfg=dict(**vars(a),GPU=gpu,initial_state_sha256=init_hash,parameters=sum(p.numel() for p in net.parameters()),
  fit_points=len(train.common),monitor_points=len(monitor.common),seed=103041,sample_seed=103073,
  method='FrozenC82 plus8member512numeric residual;235features,identicalweights.',
  labels='Y-C82FIT prediction, normalized/20. Masks loss/score only; inclusion weights loss only.',
  precision_ablation='local:common169plusconstant66;terrain:common169plus66terrainmean/std/PCA64. PCAfitonly5508FIT heightmaps, no labels.',
  selection='Min jointmean monitor bandRMSE including unchangedC82. Fixed2000updates; no confirmation choice.',
  official_val_used=False)
 atomic_json(out/'config.json',cfg)
 initial=score(net,monitor);atomic_json(out/'initial.json',initial)
 best=float(np.mean(initial['rmse']));beststep=0
 torch.save(dict(state=net.state_dict(),step=0,config=cfg),out/'best.pt')
 history=[];gen=torch.Generator(device='cuda').manual_seed(103073);train_x=None
 print('INITIAL',a.mode,initial['rmse'],'loaded',time.monotonic()-t,flush=True)
 for step in range(1,a.steps+1):
  ix=torch.randint(len(train.common),(a.batch,),device='cuda',generator=gen);x=train.x(ix)
  target=(train.target[ix,:2]-train.base[ix])/20;m=train.target[ix,2:4];weight=train.weight[ix,None]
  warm=min(1.,step/100);lr=.0005*warm*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/a.steps)))
  for pg in opt.param_groups:pg['lr']=lr
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',torch.bfloat16):pred=net(x)
  e=(pred-target[:,None]).square();wm=m*weight
  loss=(e*wm[:,None]).sum((0,1))/(8*wm.sum(0)).clamp_min(1);loss=loss.mean()
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
  assert torch.isfinite(loss)
  if step%250==0 or step==a.steps:
   val=score(net,monitor);criterion=float(np.mean(val['rmse']))
   if criterion<best:
    best=criterion;beststep=step;torch.save(dict(state=net.state_dict(),step=step,config=cfg),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
   row=dict(step=step,monitor=val,gain_vs_C82=(np.array(initial['rmse'])-val['rmse']).tolist(),best_step=beststep,loss=float(loss),seconds=time.monotonic()-t)
   history.append(row);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',row);print(a.mode,json.dumps(row),flush=True)
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);val=score(net,monitor)
 atomic_json(out/'result.json',dict(status='complete',best_step=beststep,last_step=a.steps,monitor=val,
  gain_vs_C82=(np.array(initial['rmse'])-val['rmse']).tolist(),seconds=time.monotonic()-t,official_val_used=False))
 print('DONE',a.mode,beststep,val['rmse'],flush=True)
if __name__=='__main__':main()
