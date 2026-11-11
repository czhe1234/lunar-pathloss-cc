"""C129 frozen-backbone least-squares head and full FIT audit."""
import sys,json,time,hashlib,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c127_train import create,reader,Panel,group_batch,weights
A=R/'audit/c129_fitgap'

class Features:
 def __init__(self,net):
  self.val={}
  self.hooks=[net.tx.head.register_forward_pre_hook(lambda m,a:self.val.update(tx=a[0])),net.scene.head.register_forward_pre_hook(lambda m,a:self.val.update(scene=a[0]))]
 def get(self,inverse):
  return torch.cat([self.val['tx'],self.val['scene'][inverse]],1)
 def close(self):
  for h in self.hooks:h.remove()

@torch.inference_mode()
def evaluate(net,data,route,weight=None,save=None):
 net.eval();stats=[];arr=np.lib.format.open_memmap(save,'w+',dtype=np.float32,shape=(len(data.ids),2,256,256)) if save else None
 for start in range(0,len(data.ids),16):
  ix=list(range(start,min(start+16,len(data.ids))));idx=torch.tensor(ix,device='cuda');x,prior,y,v=data.batch(idx);base=data.reference[idx]
  _,first,inverse=group_batch(data,ix);ref=base if route=='residual' else prior
  with torch.autocast('cuda',torch.bfloat16):s,t=net.components(x,first,inverse)
  p=ref+s+t;e=torch.where(v,p-y,0);tail=v&((y-base).abs()>=5)
  ss=e.double().square().sum((2,3));nn=v.sum((2,3));ts=torch.where(tail,e.double().square(),0).sum((2,3));tn=tail.sum((2,3))
  stats.append(torch.cat([ss,nn,ts,tn],1).cpu().numpy())
  if arr is not None:arr[start:start+len(ix)]=p.float().cpu().numpy()
 if arr is not None:arr.flush()
 stat=np.concatenate(stats);z=stat.sum(0);out=dict(RMSE=np.sqrt(z[:2]/z[2:4]).tolist(),valid=z[2:4].tolist(),tail_RMSE=np.sqrt(z[4:6]/z[6:8]).tolist())
 if weight is not None:
  zz=(stat*weight[:,None]).sum(0);out['weighted_RMSE']=np.sqrt(zz[:2]/zz[2:4]).tolist()
 return out,stat

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');start=time.monotonic()
 plan=json.loads((R/'audit/c127_joint/protocol.json').read_text());rr=reader()
 ids=plan['fit'][:8] if args.smoke else plan['fit']
 data=Panel(rr,ids,reference=True)
 # Original FIT inclusion weights, not subset frequency recomputation in smoke.
 fullids=plan['fit'];from collections import Counter
 allfit=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(rr.rows[int(i)]['terrain_id'] for i in allfit);sel=Counter(rr.rows[int(i)]['terrain_id'] for i in fullids)
 w=torch.tensor([cnt[rr.rows[i]['terrain_id']]/sel[rr.rows[i]['terrain_id']] for i in ids],device='cuda',dtype=torch.float64);w/=w.mean()
 mid=np.load(R/'audit/c101_residual/fit/mapid.npy',mmap_mode='r').reshape(-1,1536)
 px=np.load(R/'audit/c101_residual/fit/pixel.npy',mmap_mode='r').reshape(-1,1536)
 assert np.all(mid==mid[:,:1]);lookup={int(v):i for i,v in enumerate(mid[:,0])}
 pix=torch.tensor(np.stack([px[lookup[i]] for i in ids]),device='cuda',dtype=torch.long)
 net=create('residual');ck=torch.load(R/'runs/c127_residual_independent/best.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);net=net.cuda().to(memory_format=torch.channels_last);net.eval()
 hooks=Features(net)
 G=torch.zeros(2,73,73,device='cuda',dtype=torch.float64);q=torch.zeros(2,73,device='cuda',dtype=torch.float64);yy=torch.zeros(2,device='cuda',dtype=torch.float64)
 old_sample_s=torch.zeros(2,device='cuda',dtype=torch.float64);sample_n=old_sample_s.clone();max_replay=0.;stats=[]
 arr=None if args.smoke else np.lib.format.open_memmap(A/'fit_parent_prediction.npy','w+',dtype=np.float32,shape=(len(ids),2,256,256))
 with torch.inference_mode():
  for st in range(0,len(ids),16):
   ix=list(range(st,min(st+16,len(ids))));idx=torch.tensor(ix,device='cuda');x,_,y,v=data.batch(idx);base=data.reference[idx];_,first,inverse=group_batch(data,ix)
   with torch.autocast('cuda',torch.bfloat16):s,t=net.components(x,first,inverse)
   pred=base+s+t;err=torch.where(v,pred-y,0);tail=v&((y-base).abs()>=5)
   stats.append(torch.cat([err.double().square().sum((2,3)),v.sum((2,3)),torch.where(tail,err.double().square(),0).sum((2,3)),tail.sum((2,3))],1).cpu().numpy())
   if arr is not None:arr[st:st+len(ix)]=pred.float().cpu().numpy()
   feature=hooks.get(inverse).flatten(2);pp=pix[idx]
   feature=feature.gather(2,pp[:,None].expand(-1,72,-1)).transpose(1,2).reshape(-1,72).double()
   feature=torch.cat([feature,torch.ones(len(feature),1,device='cuda',dtype=torch.float64)],1)
   residual=(y-base).flatten(2).gather(2,pp[:,None].expand(-1,2,-1)).transpose(1,2).reshape(-1,2).double()
   valid=v.flatten(2).gather(2,pp[:,None].expand(-1,2,-1)).transpose(1,2).reshape(-1,2)
   oldpred=(s+t).flatten(2).gather(2,pp[:,None].expand(-1,2,-1)).transpose(1,2).reshape(-1,2).double()
   theta=torch.cat([net.tx.head.weight[:,:,0,0]*4,net.scene.head.weight[:,:,0,0]*4,4*(net.tx.head.bias+net.scene.head.bias)[:,None]],1).double()
   max_replay=max(max_replay,float((feature@theta.T-oldpred).abs().max()))
   ww=w[idx,None].expand(-1,1536).reshape(-1)
   for b in range(2):
    wb=ww*valid[:,b];r=torch.where(valid[:,b],residual[:,b],0);G[b]+=feature.T@(feature*wb[:,None]);q[b]+=feature.T@(r*wb);yy[b]+=(r.square()*wb).sum()
    old_sample_s[b]+=(torch.where(valid[:,b],oldpred[:,b]-residual[:,b],0).square()*ww).sum();sample_n[b]+=wb.sum()
   if (st//16)%40==0:print('COVARIANCE',st+len(ix),len(ids),round(time.monotonic()-start,1),flush=True)
 if arr is not None:arr.flush()
 hooks.close();assert max_replay<1e-3,max_replay
 gg=G.cpu().numpy();qq=q.cpu().numpy();yyy=yy.cpu().numpy();theta=[];details=[]
 for b in range(2):
  count=gg[b,-1,-1];mx=gg[b,-1,:-1]/count;my=qq[b,-1]/count
  cov=gg[b,:-1,:-1]/count-np.outer(mx,mx);cy=qq[b,:-1]/count-mx*my
  std=np.sqrt(np.maximum(np.diag(cov),1e-16));C=cov/std[:,None]/std[None,:];u=cy/std
  ev,vec=np.linalg.eigh((C+C.T)/2);ridge=1e-6*max(float(ev[-1]),1.)
  beta=(vec@((vec.T@u)/(np.maximum(ev,0)+ridge)))/std
  theta.append(np.r_[beta,my-beta@mx])
  details.append(dict(ridge=ridge,min_eigen=float(ev[0]),max_eigen=float(ev[-1]),effective_rank=int((ev>ev[-1]*1e-8).sum())))
 theta=np.stack(theta)
 opt_s=np.array([theta[b]@gg[b]@theta[b]-2*theta[b]@qq[b]+yyy[b] for b in range(2)])
 summary=dict(GPU=gpu,features=72,intercept=True,FIT_maps=len(ids),FIT_points=len(ids)*1536,head_optimization='closed form standardized ridge,lambda=1e-6*maxeigen,no validation tuning',
  old_sample_weighted_RMSE=(old_sample_s/sample_n).sqrt().tolist(),closed_sample_weighted_RMSE=np.sqrt(np.maximum(opt_s,0)/gg[:,-1,-1]).tolist(),
  max_original_head_replay_error=max_replay,eigen=details,official_val_used=False)
 if args.smoke:
  summary['status']='passed';atomic_json(A/'readout_smoke.json',summary);print('SMOKE',json.dumps(summary),flush=True);return
 assert (A/'readout_smoke.json').exists()
 np.savez(A/'normal_equations.npz',G=gg,q=qq,yy=yyy,theta=theta)
 stat=np.concatenate(stats);np.save(A/'fit_parent_by_map.npy',stat);z=stat.sum(0);zw=(stat*w.cpu().numpy()[:,None]).sum(0)
 summary['parent_fullFIT']=dict(RMSE=np.sqrt(z[:2]/z[2:4]).tolist(),weighted_RMSE=np.sqrt(zw[:2]/zw[2:4]).tolist(),tail_RMSE=np.sqrt(z[4:6]/z[6:8]).tolist())
 with torch.no_grad():
  net.tx.head.weight.copy_(torch.tensor(theta[:,:48,None,None]/4,device='cuda',dtype=torch.float32))
  net.scene.head.weight.copy_(torch.tensor(theta[:,48:72,None,None]/4,device='cuda',dtype=torch.float32))
  net.tx.head.bias.copy_(torch.tensor(theta[:,72]/4,device='cuda',dtype=torch.float32));net.scene.head.bias.zero_()
 torch.save(dict(state=net.state_dict(),parent='c127_residual_independent best',head_method=summary['head_optimization']),A/'readout.pt')
 v,stat=evaluate(net,data,'residual',w.cpu().numpy(),A/'fit_readout_prediction.npy');summary['closed_fullFIT']=v;np.save(A/'fit_readout_by_map.npy',stat)
 monitor=Panel(rr,plan['monitor'],reference=True)
 v,stat=evaluate(net,monitor,'residual',save=A/'monitor_readout_prediction.npy');summary['closed_monitor']=v;np.save(A/'monitor_readout_by_map.npy',stat)
 for name,route,where in [('residual_last','residual','runs/c127_residual_independent/last.pt'),('direct_joint_best','direct','runs/c127_direct_joint/best.pt'),('direct_joint_last','direct','runs/c127_direct_joint/last.pt')]:
  ck=torch.load(R/where,map_location='cpu',weights_only=False);net.load_state_dict(ck['state'])
  v,stat=evaluate(net,data,route,w.cpu().numpy());summary[name+'_fullFIT']=v;np.save(A/('fit_'+name+'_by_map.npy'),stat)
  print('FULLFIT',name,v,flush=True)
 summary['status']='complete';summary['seconds']=time.monotonic()-start
 atomic_json(A/'readout.json',summary);print('DONE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
