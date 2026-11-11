import sys,json,time,math,os,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from collections import Counter
from src.c117_factorized import Field
from src.c105_context import reader,ResidualPanel,digest
from src.c108_train import evaluate
A=R/'audit/c118_location'
class AuxField(Field):
 def __init__(self):
  super().__init__('factorized');self.aux=nn.Conv2d(96,6,1)
 def outputs(self,x):
  b=len(x);r=(-128*x[:,6,0,0]).round().long().clamp(0,255);c=(-128*x[:,5,0,0]).round().long().clamp(0,255)
  z=self.encode(x);source=self.source(z[torch.arange(b,device=x.device),:,r,c])[:,:,None,None]
  rx=self.receiver(z);q=self.query(x.clamp(-12,12));interaction=rx*torch.tanh(source)
  v=self.mix(torch.cat([z,source.expand(-1,-1,256,256),q,interaction],1));v=v+self.readout(v)
  return 4*self.head(v),self.aux(v).reshape(b,2,3,256,256)
 def forward(self,x):return self.outputs(x)[0]
def create(parent):
 torch.manual_seed(118041);net=AuxField();ck=torch.load(parent,map_location='cpu',weights_only=False)
 missing,unexpected=net.load_state_dict(ck['state'],strict=False);assert set(missing)=={'aux.weight','aux.bias'} and not unexpected
 return net
def losses(net,data,ix,weight,prior):
 x,b,y,w=data.batch(ix)
 with torch.autocast('cuda',torch.bfloat16):delta,logit=net.outputs(x)
 e=torch.where(w,b+delta.float()-y,0.);ww=weight[ix,None,None,None];s=(e.square()*ww).sum((0,2,3));c=(w*ww).sum((0,2,3))
 mse=(s/c).mean()/25
 residual=y-b;cls=torch.where(residual>=5,2,torch.where(residual<=-5,0,1)).long()
 ce=F.cross_entropy(logit.float().reshape(-1,3,256,256),cls.reshape(-1,256,256),reduction='none').reshape(len(ix),2,256,256)
 pri=prior[torch.arange(2,device='cuda')[None,:,None,None],cls]
 ce=(ce*w*ww/(3*pri.clamp_min(1e-6))).sum((0,2,3))/c
 return mse,ce.mean(),s,c
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['plain','location'],default='location');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());old=json.loads((R/'audit/c105_context/protocol.json').read_text())
 net=create(R/plan['parent']);ih=digest(net);net=net.cuda().to(memory_format=torch.channels_last)
 r=reader();train=ResidualPanel(r,old['fit'][:16] if a.smoke else old['fit'],'context')
 allfit=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(r.rows[int(i)]['terrain_id'] for i in allfit);sel=Counter(train.terrain)
 weight=torch.tensor([cnt[t]/sel[t] for t in train.terrain],device='cuda');weight/=weight.mean()
 prior=torch.zeros(2,3,device='cuda',dtype=torch.float64)
 for k in range(0,len(train.ids),64):
  res=train.y[k:k+64]-train.reference[k:k+64];valid=train.w[k:k+64];cls=torch.where(res>=5,2,torch.where(res<=-5,0,1));ww=weight[k:k+64,None,None,None]
  for c in range(3):prior[:,c]+=((cls==c)*valid*ww).sum((0,2,3),dtype=torch.float64)
 prior=(prior/prior.sum(1,keepdim=True)).float()
 if a.smoke:
  ix=torch.arange(8,device='cuda');mse,ce,_,_=losses(net,train,ix,weight,prior)
  params=[p for n,p in net.named_parameters() if not n.startswith(('head.','aux.'))]
  norm=[]
  for loss in [mse,ce]:
   gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
   norm.append(float(torch.stack([v.float().square().sum() for v in gg if v is not None]).sum().sqrt()))
  opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True);opt.zero_grad(set_to_none=True);(mse+.25*ce).backward();opt.step();assert torch.isfinite(mse+ce)
  atomic_json(A/'smoke.json',dict(GPU=g,initial_sha=ih,parameters=sum(p.numel() for p in net.parameters()),backbone_gradient_norm_MSE_CE=norm,fit_prior=prior.tolist(),losses=[float(mse),float(ce)]));print('SMOKE',norm,flush=True);return
 assert ih==plan['initial_sha256'];out=R/'runs'/('c118_'+a.mode);out.mkdir(exist_ok=False);monitor=ResidualPanel(r,old['monitor'],'context')
 cfg=dict(mode=a.mode,GPU=g,initial_sha256=ih,parent=plan['parent'],prior=prior.tolist(),batch=8,epochs=8,aux_weight=.25 if a.mode=='location' else 0.,official_val_used=False)
 atomic_json(out/'config.json',cfg);opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True)
 start=time.monotonic();base=evaluate(net,monitor,8,force_base=True);init=evaluate(net,monitor,8);be=0;best=min(np.mean(init['rmse']),np.mean(base['rmse']))
 initial_kind='parent' if np.mean(init['rmse'])<np.mean(base['rmse']) else 'C82'
 torch.save(dict(state=net.state_dict(),step=0,epoch=0,kind=initial_kind,config=cfg),out/'best.pt')
 atomic_json(out/'initial.json',dict(base=base,parent=init,kind=initial_kind));history=[]
 gen=torch.Generator().manual_seed(118073);steps=math.ceil(len(train.ids)/8)
 for ep in range(1,9):
  net.train();order=torch.randperm(len(train.ids),generator=gen).cuda();ss=torch.zeros(4,device='cuda',dtype=torch.float64);auxsum=0.
  for j,ix in enumerate(order.split(8)):
   step=(ep-1)*steps+j+1;lr=.0004*min(1.,.1+.9*step/100)*(.15+.85*.5*(1+math.cos(math.pi*(step-1)/(8*steps))))
   for pg in opt.param_groups:pg['lr']=lr
   mse,ce,s,c=losses(net,train,ix,weight,prior);loss=mse+cfg['aux_weight']*ce
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   ss+=torch.cat([s.detach(),c]);auxsum+=float(ce.detach())
  val=evaluate(net,monitor,8);crit=np.mean(val['rmse'])
  if crit<best:
   best=float(crit);be=ep;torch.save(dict(state=net.state_dict(),step=step,epoch=ep,kind='model',config=cfg),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  rec=dict(epoch=ep,step=step,fit_running_rmse=(ss[:2]/ss[2:]).sqrt().tolist(),balancedCE=auxsum/steps,monitor=val,gain=(np.array(base['rmse'])-val['rmse']).tolist(),best_epoch=be,seconds=time.monotonic()-start)
  history.append(rec);atomic_json(out/'history.json',history);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(rec),flush=True)
  if ep>=4 and ep-be>=3:break
 torch.save(dict(state=net.state_dict(),step=step,epoch=ep,kind='model',config=cfg),out/'last.pt')
 atomic_json(out/'result.json',dict(status='complete',best_epoch=be,last_epoch=ep,seconds=time.monotonic()-start,official_val_used=False))
 print('DONE',a.mode,be,ep,flush=True)
if __name__=='__main__':main()
