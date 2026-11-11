"""Label-assisted native-state feasibility diagnostic. This is NOT a predictor."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c136_train import Dataset,score
from src.c136_model import NativeState
import numpy as np,torch
A=R/'audit/c137_oracle'
class FreeState(torch.nn.Module):
 def __init__(self,logs,dlogs):
  super().__init__();self.raw=torch.nn.Parameter(torch.atanh((logs/8).clamp(-.99999,.99999))*8)
  self.draw=torch.nn.Parameter(torch.atanh((dlogs/8).clamp(-.99999,.99999))*8)
 def forward(self,g,base):
  logs=8*torch.tanh(self.raw/8);dl=8*torch.tanh(self.draw/8)
  sp=logs[:,:2].exp();q=logs[:,2:].reshape(-1,2).exp()
  si=g['si'].long();ai=g['ai'].long()
  z=g['dw']*dl.exp();z=z.index_add(0,si[:,0],g['sw']*sp[si[:,1]])
  z=z.index_add(0,ai[:,0],g['aw']*q[ai[:,1]])
  ratio=(z[g['inv'].long()]*g['pw']).sum(1);ratio=torch.where(g['active'],ratio,torch.ones_like(ratio))
  return base-10/2.302585092994046*ratio.clamp_min(1e-12).log()
@torch.no_grad()
def evaluate(states,data):
 ss=np.zeros(2);nn=np.zeros(2);tailss=np.zeros(2);tailnn=np.zeros(2);sign=np.zeros(2)
 for j,m in enumerate(states):
  p=m(data.graph[j],data.base[j]);v=data.v[j];e=torch.where(v,p-data.y[j],0.).double();r=data.y[j]-data.base[j];tail=v&(r.abs()>=5)
  ss+=e.square().sum(0).cpu().numpy();nn+=v.sum(0).cpu().numpy()
  tailss+=(e.square()*tail).sum(0).cpu().numpy();tailnn+=tail.sum(0).cpu().numpy();sign+=(((p-data.base[j])*r>0)&tail).sum(0).cpu().numpy()
 return dict(RMSE=np.sqrt(ss/nn).tolist(),valid_count=nn.tolist(),tail_RMSE=np.sqrt(tailss/tailnn).tolist(),tail_sign_accuracy=(sign/tailnn).tolist())
def main():
 gpu=gpu_guard();torch.set_num_threads(3);torch.set_float32_matmul_precision('high');torch.manual_seed(137041)
 ids=json.loads((R/'audit/c136_native/protocol.json').read_text())['fit'][:16];data=Dataset('fit',ids);data.w.fill_(1);data.den=data.v.float().mean((0,1))
 net=NativeState('native').cuda();ck=torch.load(R/'runs/c136_capacity16/best.pt',map_location='cuda',weights_only=False);net.load_state_dict(ck['state']);net.eval()
 states=torch.nn.ModuleList()
 with torch.no_grad():
  enc=net.make_context(data.h)
  for j,g in enumerate(data.graph):
   e=enc[data.gid[j]];face=e.reshape(16,-1).T[net.faces].float().mean(1)
   tx=torch.nn.functional.grid_sample(e[None].float(),g['txxy'].reshape(1,1,1,2),align_corners=True).reshape(16)
   logs=net.state(torch.cat((g['feat'].float(),face,tx.expand(len(face),16)),1))
   ix=g['uf'].long();dlog=net.measure(torch.cat((g['df'].float(),face[ix],tx.expand(len(ix),16)),1))
   states.append(FreeState(logs,dlog))
 initial=evaluate(states,data);old=score(net,data)[0]
 assert max(abs(x-y) for x,y in zip(initial['RMSE'],old['RMSE']))<1e-4,(initial,old)
 del net,enc
 opt=torch.optim.Adam(states.parameters(),lr=.03,foreach=True)
 start=time.monotonic();hist=[];rng=np.random.default_rng(137043);order=[];best=max(initial['RMSE'])
 for step in range(1,1001):
  if not order:order=rng.permutation(16).tolist()
  ix=order[:4];order=order[4:]
  for pg in opt.param_groups:pg['lr']=.003+.027*(1+np.cos(np.pi*(step-1)/1000))/2
  loss=0
  for j in ix:
   p=states[j](data.graph[j],data.base[j]);loss+=(((p-data.y[j]).square()*data.v[j]).mean(0)/data.den).mean()/100
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(states.parameters(),5);opt.step();assert torch.isfinite(loss)
  if step%100==0:
   result=evaluate(states,data);rec=dict(step=step,seconds=time.monotonic()-start,training= result);hist.append(rec);atomic_json(A/'progress.json',rec);print(json.dumps(rec),flush=True)
   if max(result['RMSE'])<best:
    best=max(result['RMSE']);torch.save(dict(states=states.state_dict(),ids=ids,step=step,labels_used_directly=True),R/'runs/c137_oracle16.pt')
   if max(result['RMSE'])<.5:break
 out=dict(status='complete',GPU=gpu,initial=initial,final=hist[-1],history=hist,ids=ids,
  labels_used_directly=True,learned_H_TX_mapping=False,official_VAL_used=False,
  interpretation='Diagnostic only. Per-map independent native power states use training labels directly. A low error establishes expressibility of the fixed positive transport with these calibration factors, not predictability from H/TX or competition progress.')
 atomic_json(A/'result.json',out);print('DONE',json.dumps(out),flush=True)
if __name__=='__main__':main()
