import sys,json,time,math,os,argparse,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from torch.nn import functional as F
from collections import Counter
from src.c118_location import AuxField
from src.c105_context import reader,ResidualPanel,digest
from src.c108_train import evaluate
A=R/'audit/c119_polarity'
def pair_moments(error,pos,neg,weight):
 # Every ordered positive-to-negative edge within a 5x5 neighborhood is counted once.
 pool=lambda z:F.avg_pool2d(z,5,1,2,divisor_override=1)
 nn=pool(neg.float());ne=pool(error*neg);ne2=pool(error.square()*neg)
 pp=pos*weight[:,None,None,None]
 den=(pp*nn).sum((0,2,3)).clamp_min(1)
 endpoint=(pp*(error.square()*nn+ne2)).sum((0,2,3))/den
 cross=(pp*(2*error*ne)).sum((0,2,3))/den
 return endpoint,(endpoint-cross).clamp_min(0),den
def create(parent):
 torch.manual_seed(119041);net=AuxField();ck=torch.load(parent,map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);return net
def loss_parts(net,data,ix,weight):
 x,b,y,w=data.batch(ix)
 with torch.autocast('cuda',torch.bfloat16):p=b+net(x).float()
 error=torch.where(w,p-y,0.);ww=weight[ix,None,None,None];s=(error.square()*ww).sum((0,2,3));c=(w*ww).sum((0,2,3));main=(s/c).mean()/25
 residual=y-b;pos=w&(residual>=5);neg=w&(residual<=-5)
 endpoint,pair,den=pair_moments(error,pos,neg,weight[ix])
 return main,endpoint.mean()/25,pair.mean()/25,s,c,den
def check_moments():
 torch.manual_seed(119097);e=torch.randn(2,2,4,4,device='cuda',requires_grad=True)
 state=torch.randint(3,e.shape,device='cuda');pos=state==2;neg=state==0;w=torch.tensor([1.,2.],device='cuda')
 ep,pair,den=pair_moments(e,pos,neg,w);direct=[]
 for b in range(2):
  se=sp=ct=0.
  for j in range(2):
   for r in range(4):
    for c in range(4):
     if not pos[j,b,r,c]:continue
     for rr in range(max(0,r-2),min(4,r+3)):
      for cc in range(max(0,c-2),min(4,c+3)):
       if not neg[j,b,rr,cc]:continue
       se=se+w[j]*(e[j,b,r,c]**2+e[j,b,rr,cc]**2);sp=sp+w[j]*(e[j,b,r,c]-e[j,b,rr,cc])**2;ct=ct+w[j]
  direct.append(torch.stack([se/ct,sp/ct]))
 direct=torch.stack(direct)
 assert torch.allclose(ep,direct[:,0],atol=2e-6) and torch.allclose(pair,direct[:,1],atol=2e-6)
 g1=torch.autograd.grad(pair.sum(),e,retain_graph=True)[0];g2=torch.autograd.grad(direct[:,1].sum(),e)[0]
 assert torch.allclose(g1,g2,atol=1e-6)
 common=torch.ones_like(e)*3;_,p,_=pair_moments(common,pos,neg,w);assert float(p.max())<1e-6
 return dict(explicit_pair_value_max_error=float((pair-direct[:,1]).abs().max()),gradient_max_error=float((g1-g2).abs().max()),common_error_cancelled=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['plain','endpoint','pair'],default='pair');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
 plan=json.loads((A/'protocol.json').read_text());assert hashlib.sha256((R/plan['parent']).read_bytes()).hexdigest()==plan['parent_sha256']
 net=create(R/plan['parent']);ih=digest(net);net=net.cuda().to(memory_format=torch.channels_last)
 old=json.loads((R/'audit/c105_context/protocol.json').read_text());r=reader();train=ResidualPanel(r,old['fit'][:16] if a.smoke else old['fit'],'context')
 full=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(r.rows[int(i)]['terrain_id'] for i in full);sel=Counter(train.terrain)
 weight=torch.tensor([cnt[t]/sel[t] for t in train.terrain],device='cuda');weight/=weight.mean()
 if a.smoke:
  ck=check_moments();ix=torch.arange(8,device='cuda');m,ep,p,_,_,den=loss_parts(net,train,ix,weight)
  opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True);opt.zero_grad(set_to_none=True);(m+.02*p).backward();opt.step();assert torch.isfinite(m+ep+p)
  atomic_json(A/'smoke.json',dict(GPU=gpu,initial_sha256=ih,checks=ck,loss_main_endpoint_pair=[float(m),float(ep),float(p)],weighted_pair_count=den.tolist(),parameters=sum(p.numel() for p in net.parameters())))
  print('SMOKE',ck,[float(m),float(ep),float(p)],flush=True);return
 assert ih==plan['initial_sha256']
 out=R/'runs'/('c119_'+a.mode);out.mkdir(exist_ok=False);monitor=ResidualPanel(r,old['monitor'],'context')
 cfg=dict(mode=a.mode,GPU=gpu,initial_sha256=ih,parent=plan['parent'],parent_sha256=plan['parent_sha256'],batch=8,epochs=8,pair_radius=2,lambda_aux=.02,official_val_used=False)
 atomic_json(out/'config.json',cfg);opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=.01,fused=True)
 start=time.monotonic();base=evaluate(net,monitor,8,force_base=True);init=evaluate(net,monitor,8);best=float(min(np.mean(init['rmse']),np.mean(base['rmse'])));be=0
 kind='parent' if np.mean(init['rmse'])<np.mean(base['rmse']) else 'C82'
 torch.save(dict(state=net.state_dict(),step=0,epoch=0,kind=kind,config=cfg),out/'best.pt')
 atomic_json(out/'initial.json',dict(base=base,parent=init,kind=kind))
 gen=torch.Generator().manual_seed(119073);steps=math.ceil(len(train.ids)/8);hist=[]
 for epoch in range(1,9):
  net.train();order=torch.randperm(len(train.ids),generator=gen).cuda();ss=torch.zeros(4,device='cuda',dtype=torch.float64);saux=torch.zeros(2,device='cuda')
  for j,ix in enumerate(order.split(8)):
   step=(epoch-1)*steps+j+1;lr=.0004*min(1.,.1+.9*step/100)*(.15+.85*.5*(1+math.cos(math.pi*(step-1)/(8*steps))))
   for pg in opt.param_groups:pg['lr']=lr
   main,endpoint,pair,s,c,_=loss_parts(net,train,ix,weight)
   loss=main if a.mode=='plain' else main+.02*(endpoint if a.mode=='endpoint' else pair)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
   ss+=torch.cat([s.detach(),c]);saux+=torch.stack([endpoint.detach(),pair.detach()])
  val=evaluate(net,monitor,8);crit=float(np.mean(val['rmse']))
  if crit<best:
   best=crit;be=epoch;torch.save(dict(state=net.state_dict(),step=step,epoch=epoch,kind='model',config=cfg),out/'best.tmp');os.replace(out/'best.tmp',out/'best.pt')
  rec=dict(epoch=epoch,step=step,fit_running_rmse=(ss[:2]/ss[2:]).sqrt().tolist(),fit_endpoint_pair_loss=(saux/steps).tolist(),monitor=val,gain=(np.array(base['rmse'])-val['rmse']).tolist(),best_epoch=be,seconds=time.monotonic()-start)
  hist.append(rec);atomic_json(out/'history.json',hist);atomic_json(out/'progress.json',rec);print('EPOCH',json.dumps(rec),flush=True)
  if epoch>=4 and epoch-be>=3:break
 torch.save(dict(state=net.state_dict(),step=step,epoch=epoch,kind='model',config=cfg),out/'last.pt')
 atomic_json(out/'result.json',dict(status='complete',best_epoch=be,last_epoch=epoch,seconds=time.monotonic()-start,official_val_used=False));print('DONE',a.mode,be,epoch,flush=True)
if __name__=='__main__':main()
