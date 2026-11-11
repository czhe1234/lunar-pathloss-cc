"""No-training D4 equivariance and orbit-average screen on fixed sampled DEV."""
from pathlib import Path
import sys,time,json,csv
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
from src.cc_core import gpu_guard
gpu=gpu_guard();torch.cuda.set_per_process_memory_fraction(.15);torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
from src.c41_d4 import transform
from c34_inference import Predictor
O=R/'audit/c41_d4_20261101';O.mkdir(exist_ok=False)
model=Predictor(R/'runs/c37_wide_periodic8')
# Recompute one complete rotated map and compare mapped features at deterministic pixels.
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
dev=set(json.loads((R/'audit/internal_split.json').read_text())['dev_terrain_ids'])
ii=next(i for i,q in enumerate(rows) if q['terrain_id'] in dev);q=rows[ii]
h=np.array(np.load(R.parent/'data/train_hm.npy',mmap_mode='r')[int(q['hm_row'])],np.float32);tr,tc=int(q['tx_row']),int(q['tx_col'])
capt=[]
hook=model.net.register_forward_pre_hook(lambda module,args:capt.append(args[0].detach().cpu().numpy().copy()))
model.predict(h,tr,tc);x0=np.concatenate(capt);capt.clear()
model.predict(np.rot90(h).copy(),255-tc,tr);x1=np.concatenate(capt);capt.clear()
hook.remove()
# x1 is ordered in rotated image; map it back to original query positions.
x1back=np.rot90(x1.reshape(256,256,294),-1,axes=(0,1)).reshape(-1,294).copy()
z=transform(torch.from_numpy(x0).cuda(),1).cpu().numpy()
delta=abs(z-x1back);by=delta.max(0);bad=np.argsort(by)[-15:]
replay={'map':ii,'worst_columns':[[int(k),float(by[k])] for k in bad[::-1]],'max_abs':float(delta.max()),'rms_all_features':float(np.sqrt(np.mean(delta.astype(float)**2))),'note':'Floating ray/surface sampling may change at angular/grid ties; comparison checks derived coordinate, derivative, patch and direction transforms.'}
# Exact coordinate/derivative/patch transformations must match to rounding.
cols=[3,4,5,6,32,33,34,35,36,37,38]+list(range(47,96))
assert np.max(delta[:,cols])<1e-4,replay
(O/'feature_replay.json').write_text(json.dumps(replay,indent=2)+'\n')
X=np.load(R/'features/c10_points/dev_x.npy',mmap_mode='r')
Y=np.load(R/'features/c6_points/dev_y.npy');F=np.load(R/'features/c6_points/dev_fspl.npy');M=np.load(R/'features/c6_points/dev_mask.npy').astype(bool)
pred=[];t0=time.time()
with torch.inference_mode():
 for k in range(8):
  out=[]
  for lo in range(0,len(X),4096):
   x=torch.tensor(np.array(X[lo:lo+4096]),device='cuda');x=transform(x,k)
   with torch.autocast('cuda',dtype=torch.bfloat16):p=model.net(x).mean(1)
   out.append(p.cpu().numpy()*20)
  pred.append(np.concatenate(out)+F)
  print(json.dumps({'rotation':k,'seconds':time.time()-t0}),flush=True)
pred=np.array(pred);metrics=[]
for name,p in [('identity',pred[0]),('rotations4',pred[:4].mean(0)),('D4',pred.mean(0))]:
 rms=[float(np.sqrt(np.mean((p[:,k][M[:,k]].astype(float)-Y[:,k][M[:,k]])**2))) for k in [0,1]]
 metrics.append({'mode':name,'RMSE':rms})
std=[float(np.sqrt(np.mean(pred[:,:,k].var(0)[M[:,k]]))) for k in [0,1]]
np.save(O/'dev_d4_prediction.npy',pred.mean(0));np.save(O/'dev_rotation_prediction.npy',pred[:4].mean(0))
out={'scope':'No training; same complete sampled DEV as C37; physical rotation screen, not official score','gpu':gpu,'metrics':metrics,'orbit_RMS_std_db':std,'fresh_feature_replay':replay,'seconds':time.time()-t0}
(O/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
