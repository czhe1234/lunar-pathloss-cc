"""Bounded diagnostic: can a common raster-height phase explain C38 errors?
Candidate phase is not an actual recovered TX coordinate. No neural training.
"""
import sys,time,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c47_train import load_teacher,teacher_net
from src.c10_phase_geometry import phase_features_gpu
from src.c47_inference import Predictor as Features
A=R/'audit/c64_height_phase_20261102'
@torch.no_grad()
def translated_features(x,delta):
 if delta==0:return x
 z=x.clone();z[:,0]+=delta/100;z[:,31]+=delta;z[:,39]+=delta;z[:,41:47]+=delta
 ht=z[:,31].double()
 for j,q in enumerate([.25,.5,1.,2.]):
  z[:,96+2*j]=torch.sin(2*np.pi*ht/q).float();z[:,97+2*j]=torch.cos(2*np.pi*ht/q).float()
 crit=2.5+2*torch.sinh(3*z[:,27]);phase=torch.arange(8,device='cuda',dtype=torch.float64)/8
 z[:,272:280]=(torch.asinh((crit.double()[:,None]-(3-torch.remainder(ht[:,None]+phase,1)))/2)/3).float()
 z[:,280:294]=phase_features_gpu(z[:,:280]);return z
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.3);torch.set_float32_matmul_precision('high')
s,c=load_teacher();net=teacher_net(s,c)
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));protocol=json.loads((R/'audit/c43_architecture_20261102/protocol.json').read_text())
fitids=protocol['fit64_ids'];fitall=np.load(R/'features/c38_dense/fit_map_indices.npy');look={int(i):j for j,i in enumerate(fitall)}
fitselect=np.concatenate([np.arange(look[i]*1024,(look[i]+1)*1024) for i in fitids])
devselect=np.arange(655872);devids=np.load(R/'features/c6_points/dev_map_indices.npy').tolist()
grid=np.arange(16)/16-.5;preds={};targets={};valids={}
t=time.time()
# Independent raw-H translation check of the algebraic feature transform.
f=object.__new__(Features);yy,xx=np.mgrid[:256,:256];f.yy=yy.astype(np.float32);f.xx=xx.astype(np.float32)
row=rows[fitids[0]];H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r');h=np.array(H[int(row['hm_row'])],np.float32);tr=int(row['tx_row']);tc=int(row['tx_col'])
orig,_=f.features(h,tr,tc);shifted,_=f.features(h+np.float32(.25),tr,tc);trans=translated_features(orig,.25)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 j=torch.arange(0,65536,16,device='cuda');d=(net(shifted[j]).mean(1)-net(trans[j]).mean(1)).float()*20
parity={'maximum_feature_difference':float((shifted-trans).abs().max()),'prediction_RMS_db':d.square().mean(0).sqrt().cpu().tolist(),'meaning':'Analytical input-translation approximation; not a recovered mesh or TX'}
assert max(parity['prediction_RMS_db'])<.03,parity
del orig,shifted,trans
for part,folder,lab,selection,ids,P in [('fit','c38_dense','c38_dense',fitselect,fitids,1024),('dev','c10_points','c6_points',devselect,devids,256)]:
 arr=np.load(R/f'features/{folder}/{part}_x.npy',mmap_mode='r');x=torch.tensor(np.array(arr[selection]),device='cuda');fs=np.array(np.load(R/f'features/{lab}/{part}_fspl.npy',mmap_mode='r')[selection]);out=[]
 for delta in grid:
  pr=[]
  with torch.inference_mode():
   z=translated_features(x,float(delta))
   for lo in range(0,len(x),4096):
    with torch.autocast('cuda',dtype=torch.bfloat16):q=net(z[lo:lo+4096]).mean(1).float()
    pr.append(q.cpu().numpy()*20)
  out.append(np.concatenate(pr)+fs);print(json.dumps({'part':part,'phase':float(delta),'seconds':time.time()-t}),flush=True)
 out=np.stack(out);out=np.concatenate((out,out.mean(0,keepdims=True)),0);preds[part]=out.reshape(17,len(ids),P,2)
 targets[part]=np.array(np.load(R/f'features/{lab}/{part}_y.npy',mmap_mode='r')[selection]).reshape(len(ids),P,2)
 valids[part]=np.array(np.load(R/f'features/{lab}/{part}_mask.npy',mmap_mode='r')[selection]).reshape(len(ids),P,2).astype(bool)
 del x
def score(pr,y,m):
 e=pr.astype(float)-y.astype(float);ss=(e*e*m).sum(tuple(range(e.ndim-1)));cc=m.sum(tuple(range(m.ndim-1)));return {'RMSE':np.sqrt(ss/cc).tolist(),'SSE':ss.tolist(),'count':cc.tolist()}
cal=slice(0,512);test=slice(512,1024)
scores=[score(p[:,cal],targets['fit'][:,cal],valids['fit'][:,cal]) for p in preds['fit']]
pick=int(np.argmin([np.mean(r['RMSE']) for r in scores]));baseline=8
global_results={part:{'C38':score(preds[part][baseline],targets[part],valids[part]),'FIT_selected_candidate':score(preds[part][pick],targets[part],valids[part]),'uniform_phase_ensemble':score(preds[part][-1],targets[part],valids[part])} for part in ['fit','dev']}
groups={}
for j,i in enumerate(fitids):groups.setdefault(rows[i]['terrain_id'],[]).append(j)
selected=[];chosen=[];yp=[];mp=[];bp=[];multi_y=[];multi_m=[];multi_p=[];multi_b=[];cross=[]
for terrain,indices in groups.items():
 ix=np.array(indices);y=targets['fit'][ix];m=valids['fit'][ix];pr=preds['fit'][:,ix]
 theta=int(np.argmin([np.mean(score(v[:,cal],y[:,cal],m[:,cal])['RMSE']) for v in pr[:16]]))
 selected.append({'terrain':terrain,'candidate':theta,'phase':float(grid[theta]),'maps':len(ix)})
 yp.append(y[:,test].reshape(-1,2));mp.append(m[:,test].reshape(-1,2));bp.append(pr[baseline,:,test].reshape(-1,2))
 # Indexed using explicit slices to retain (map,point,band) ordering.
 mpred=pr[theta][:,test].reshape(-1,2)
 chosen.append(mpred)
 if len(ix)>1:
  aidx=np.arange(0,len(ix),2);bidx=np.arange(1,len(ix),2)
  tt=int(np.argmin([np.mean(score(v[aidx],y[aidx],m[aidx])['RMSE']) for v in pr[:16]]))
  multi_y.append(y[bidx].reshape(-1,2));multi_m.append(m[bidx].reshape(-1,2));multi_p.append(pr[tt,bidx].reshape(-1,2));multi_b.append(pr[baseline,bidx].reshape(-1,2))
  cross.append({'terrain':terrain,'calibration_TX':len(aidx),'confirmation_TX':len(bidx),'phase':float(grid[tt])})
within={'C38':score(np.concatenate(bp),np.concatenate(yp),np.concatenate(mp)),'phase_chosen_on_other_pixels':score(np.concatenate(chosen),np.concatenate(yp),np.concatenate(mp)),'selected':selected}
cross_result={'C38':score(np.concatenate(multi_b),np.concatenate(multi_y),np.concatenate(multi_m)),'phase_chosen_on_other_TX':score(np.concatenate(multi_p),np.concatenate(multi_y),np.concatenate(multi_m)),'terrains':cross} if cross else None
out={'gpu':gpu,'seconds':time.time()-t,'parity':parity,'phase_grid':grid.tolist(),'FIT_selected_candidate':pick,'FIT_selected_delta':float(grid[pick]) if pick<16 else 'uniform_mean','global_FIT_selected_evaluation':global_results,'same_terrain_oracle_pixel_confirmation':within,'same_terrain_oracle_TX_confirmation':cross_result,'official_VAL_used':False,'limits':'Unobserved phase per terrain uses calibration labels and is diagnostic only; cannot be supplied as test input. The model counterfactual does not identify actual TX coordinates or the source mesh. Uniform ensemble exceeds original single-pass runtime.'}
atomic_json(A/'result.json',out);print(json.dumps(out),flush=True)
