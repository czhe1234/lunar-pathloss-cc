"""Fixed small-set control: can an inverse-rendered state target help the network learn?"""
import sys,json,time,argparse,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c136_train import Dataset,score
from src.c136_model import NativeState
import numpy as np,torch
from torch.nn import functional as F
A=R/'audit/c138_state'
class StateNet(NativeState):
 def forward(self,*args,**kwargs):
  self.parts=[];return super().forward(*args,**kwargs)
 def map_forward(self,encoded,g,base):
  face=encoded.reshape(16,-1).T[self.faces].float().mean(1)
  tx=F.grid_sample(encoded[None].float(),g['txxy'].reshape(1,1,1,2),align_corners=True).reshape(16)
  logs=self.state(torch.cat((g['feat'].float(),face,tx.expand(len(face),16)),1))
  ix=g['uf'].long();dl=self.measure(torch.cat((g['df'].float(),face[ix],tx.expand(len(ix),16)),1))
  self.parts.append((logs,dl))
  si=g['si'].long();ai=g['ai'].long();source=logs[:,:2].exp();q=logs[:,2:].reshape(-1,2).exp()
  gain=g['dw']*dl.exp();gain=gain.index_add(0,si[:,0],g['sw']*source[si[:,1]]);gain=gain.index_add(0,ai[:,0],g['aw']*q[ai[:,1]])
  ratio=(gain[g['inv'].long()]*g['pw']).sum(1);ratio=torch.where(g['active'],ratio,torch.ones_like(ratio))
  return base-10/2.302585092994046*ratio.clamp_min(1e-12).log()
@torch.no_grad()
def teachers(data,ck):
 values=[];weights=[]
 for j,g in enumerate(data.graph):
  log=8*torch.tanh(ck['states'][f'{j}.raw']/8);dl=8*torch.tanh(ck['states'][f'{j}.draw']/8);values.append((log,dl))
  inv=g['inv'].long();rw=torch.zeros(len(g['uf']),2,device='cuda').index_add_(0,inv.reshape(-1),(g['pw']*data.v[j,:,None]).reshape(-1,2))
  si=g['si'].long();ai=g['ai'].long();sw=torch.zeros(len(log),2,device='cuda').index_add_(0,si[:,1],g['sw']*rw[si[:,0]])
  qw=torch.zeros(len(log)*2,2,device='cuda').index_add_(0,ai[:,1],g['aw']*rw[ai[:,0]])
  dw=g['dw']*rw;den=data.v[j].sum(0).clamp_min(1)
  weights.append((sw/den,qw/den,dw/den))
 return values,weights
def main():
 p=argparse.ArgumentParser();p.add_argument('--aux',type=int,choices=[0,1],default=1);a=p.parse_args();gpu=gpu_guard()
 torch.set_num_threads(3);torch.set_float32_matmul_precision('high');torch.manual_seed(138041)
 ck=torch.load(R/'runs/c137_oracle16.pt',map_location='cuda',weights_only=False);data=Dataset('fit',ck['ids']);data.w.fill_(1);data.den=data.v.float().mean((0,1));teacher,weights=teachers(data,ck)
 net=StateNet('native').cuda();old=torch.load(R/'runs/c136_capacity16/best.pt',map_location='cuda',weights_only=False);net.load_state_dict(old['state'])
 sha=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in net.state_dict().values())).hexdigest()
 out=R/'runs'/f'c138_aux{a.aux}';out.mkdir(exist_ok=False);initial=score(net,data)[0];atomic_json(out/'initial.json',initial)
 atomic_json(out/'config.json',dict(initial_sha256=sha,aux=a.aux,steps=1000,batch=4,lr=.0007,GPU=gpu,teacher_labels_FIT_only=True,ids=data.ids,seed=138041))
 opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=0,fused=True);rng=np.random.default_rng(138043);order=[];history=[];start=time.monotonic();best=float('inf')
 for step in range(1,1001):
  if not order:order=rng.permutation(16).tolist()
  ix=order[:4];order=order[4:];h,g,b,iv=data.batch(ix);pred=torch.stack(net(h,g,b,iv))
  main=(((pred-data.y[ix]).square()*data.v[ix]).mean((0,1))/data.den).mean()/25
  aux=torch.zeros((),device='cuda')
  for j,(logs,dl) in zip(ix,net.parts):
   tl,td=teacher[j];sw,qw,dw=weights[j]
   aux+=(((logs[:,:2]-tl[:,:2]).square()*sw).sum(0)+((logs[:,2:].reshape(-1,2)-tl[:,2:].reshape(-1,2)).square()*qw).sum(0)+((dl-td).square()*dw).sum(0)).mean()/len(ix)*(10/math.log(10))**2/25
  loss=main+a.aux*aux
  for pg in opt.param_groups:pg['lr']=.00007+.00063*(1+np.cos(np.pi*(step-1)/1000))/2
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();assert torch.isfinite(loss)
  if step%200==0:
   result=score(net,data)[0];item=dict(step=step,seconds=time.monotonic()-start,main_loss=float(main.detach()),state_loss=float(aux.detach()),training=result)
   history.append(item);atomic_json(out/'history.json',history);print(json.dumps(item),flush=True)
   if np.mean(result['RMSE'])<best:
    best=np.mean(result['RMSE']);torch.save(dict(state=net.state_dict(),step=step,aux=a.aux),out/'best.pt')
 done=dict(status='complete',GPU=gpu,aux=a.aux,initial=initial,last=history[-1],initial_sha256=sha,seconds=time.monotonic()-start,
  scope='16 fixed FIT maps, same C136 points. Teacher uses FIT labels; network uses H/TX-derived inputs only. Not a validation or official score.')
 atomic_json(out/'done.json',done);print('DONE',json.dumps(done),flush=True)
if __name__=='__main__':main()
