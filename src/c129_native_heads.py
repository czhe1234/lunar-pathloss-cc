"""Diagnostic only: same frozen features, global/terrain/map conditional linear heads."""
import sys,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import torch,numpy as np
from src.c127_train import reader,Panel,create,group_batch
from src.c129_readout import Features
A=R/'audit/c129_fitgap'
def solve(G,q):
 # Multiple independent heads, constant regularizer; intercept not penalized.
 n=G[:,-1,-1].clamp_min(1);mx=G[:,-1,:-1]/n[:,None];my=q[:,-1]/n
 C=G[:,:-1,:-1]/n[:,None,None]-mx[:,:,None]*mx[:,None,:];u=q[:,:-1]/n[:,None]-mx*my[:,None]
 std=C.diagonal(dim1=-2,dim2=-1).clamp_min(1e-16).sqrt()
 C=C/std[:,:,None]/std[:,None,:]
 eye=torch.eye(72,device=G.device,dtype=G.dtype)[None]
 beta=torch.linalg.solve((C+C.transpose(1,2))*.5+1e-3*eye,(u/std)[:,:,None])[:,:,0]/std
 return torch.cat([beta,(my-(beta*mx).sum(1))[:,None]],1)
def main():
 gpu=gpu_guard();torch.set_num_threads(6);torch.set_float32_matmul_precision('high');start=time.monotonic()
 p=json.loads((R/'audit/c127_joint/protocol.json').read_text());rr=reader();ids=p['fit'];data=Panel(rr,ids,reference=True);n=len(ids)
 from collections import Counter
 full=np.load(R/'features/c38_dense/fit_map_indices.npy');cnt=Counter(rr.rows[int(i)]['terrain_id'] for i in full);sel=Counter(data.terrain)
 w=torch.tensor([cnt[t]/sel[t] for t in data.terrain],device='cuda',dtype=torch.float64)
 rr0,cc0=np.indices((256,256));flat=np.arange(65536).reshape(256,256)
 ordered=np.concatenate([flat[(rr0+cc0)%2==z] for z in [0,1]])
 pp=torch.tensor(ordered,device='cuda',dtype=torch.long)[None].expand(n,-1)
 net=create('residual');net.load_state_dict(torch.load(R/'runs/c127_residual_independent/best.pt',map_location='cpu',weights_only=False)['state']);net=net.cuda().to(memory_format=torch.channels_last);net.eval();hook=Features(net)
 G=torch.zeros(2,n,2,73,73,device='cuda',dtype=torch.float64);q=torch.zeros(2,n,2,73,device='cuda',dtype=torch.float64);yy=torch.zeros(2,n,2,device='cuda',dtype=torch.float64)
 with torch.inference_mode():
  for st in range(0,n,16):
   ix=list(range(st,min(st+16,n)));idx=torch.tensor(ix,device='cuda');x,_,y,v=data.batch(idx);base=data.reference[idx];_,first,inv=group_batch(data,ix)
   with torch.autocast('cuda',torch.bfloat16):net.components(x,first,inv)
   f=hook.get(inv).flatten(2).gather(2,pp[idx,None].expand(-1,72,-1)).transpose(1,2).double()
   f=torch.cat([f,torch.ones(len(ix),65536,1,device='cuda',dtype=torch.float64)],2)
   residual=(y-base).flatten(2).gather(2,pp[idx,None].expand(-1,2,-1)).transpose(1,2).double()
   valid=v.flatten(2).gather(2,pp[idx,None].expand(-1,2,-1)).transpose(1,2)
   for fold in range(2):
    sl=slice(fold*32768,(fold+1)*32768);X=f[:,sl]
    for b in range(2):
     wt=valid[:,sl,b]*w[idx,None];r=torch.where(valid[:,sl,b],residual[:,sl,b],0)
     G[fold,idx,b]=X.transpose(1,2)@(X*wt[:,:,None]);q[fold,idx,b]=(X.transpose(1,2)@(r*wt)[:,:,None])[:,:,0];yy[fold,idx,b]=(r.square()*wt).sum(1)
   if st%640==0:print('LOCAL_STATS',st,n,round(time.monotonic()-start,1),flush=True)
 hook.close();terrains=sorted(set(data.terrain));lookup={t:j for j,t in enumerate(terrains)};ti=torch.tensor([lookup[t] for t in data.terrain],device='cuda')
 old=torch.cat([4*net.tx.head.weight[:,:,0,0],4*net.scene.head.weight[:,:,0,0],4*(net.tx.head.bias+net.scene.head.bias)[:,None]],1).double()
 theta_old=old[None].expand(n,-1,-1)
 theta_map=torch.stack([solve(G[0,:,b],q[0,:,b]) for b in range(2)],1)
 gt=torch.zeros(len(terrains),2,73,73,device='cuda',dtype=torch.float64);qt=torch.zeros(len(terrains),2,73,device='cuda',dtype=torch.float64)
 gt.index_add_(0,ti,G[0]);qt.index_add_(0,ti,q[0]);theta_terrain=torch.stack([solve(gt[:,b],qt[:,b]) for b in range(2)],1)[ti]
 theta_global=torch.stack([solve(G[0,:,b].sum(0,keepdim=True),q[0,:,b].sum(0,keepdim=True))[0] for b in range(2)])[None].expand(n,-1,-1)
 result=dict(scope='diagnostic: original72frozenfeatures; 32768checkerboardFITpixels/map fit new heads; complementary32768 score heads. Parent backbone previously trained on all nativeFITpixels, so this is not independent unseen-data evaluation.',
  GPU=gpu,models={},official_val_used=False,regularization='fixed1e-3standardizedridge,no validation tuning',
  meaning='terrain/map heads use TRAIN-label-derived coefficients and cannot be deployed to unseen terrain/map; diagnose conditional readability only')
 for name,theta in [('original',theta_old),('global',theta_global),('terrain_oracle',theta_terrain),('map_oracle',theta_map)]:
  results={}
  for fold,label in [(0,'head_fit'),(1,'head_check')]:
   s=torch.einsum('nbi,nbij,nbj->nb',theta,G[fold],theta)-2*(theta*q[fold]).sum(2)+yy[fold]
   den=G[fold,:,:,-1,-1];ordinary=(s/w[:,None]).sum(0)/(den/w[:,None]).sum(0);weighted=s.sum(0)/den.sum(0)
   results[label]=dict(RMSE=ordinary.clamp_min(0).sqrt().tolist(),weighted_RMSE=weighted.clamp_min(0).sqrt().tolist(),valid=(den/w[:,None]).sum(0).tolist())
  result['models'][name]=results
 result['seconds']=time.monotonic()-start;result['status']='complete'
 np.savez(A/'conditional_native_head_stats.npz',G=G.cpu().numpy(),q=q.cpu().numpy(),yy=yy.cpu().numpy(),map_ids=np.array(ids),terrain_index=ti.cpu().numpy())
 atomic_json(A/'conditional_native_heads.json',result);print('DONE',json.dumps(result),flush=True)
if __name__=='__main__':main()
