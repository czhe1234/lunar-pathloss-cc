"""C54 compile/fusion speed check; two different FIT terrains catch stale inputs."""
import sys,time,csv,json,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c54_inference import Predictor
from src.c3_geometry import geometry
class FastPredictor(Predictor):
 @torch.inference_mode()
 def set_terrain(self,H,tr,tc):
  h=torch.from_numpy(np.array(H,np.float32,copy=True)).cuda();sy,sx=torch.gradient(h,dim=(-2,-1))
  self.high.fields.copy_(torch.stack((h,sx,sy),-1)[None])
  self.high.map_info.copy_(torch.tensor([[0,int(tr),int(tc)]],device='cuda',dtype=torch.long))
  g=geometry(h[None],torch.tensor([tr],device='cuda'),torch.tensor([tc],device='cuda'))[0,:8].float()
  t=(torch.arange(8,device='cuda',dtype=torch.float32)+.5)/8;b=2*torch.sinh(3*g)
  self.high.margins.copy_(torch.stack(((b+.95*t[:,None,None]).amax(0),(b+.95*t[:,None,None]+(1-t[:,None,None])).amax(0)),-1).half()[None])
gpu=gpu_guard();torch.set_num_threads(8);net=FastPredictor(R/'runs/c54_combo_dev')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
inputs=[(i,np.array(H[int(rows[i]['hm_row'])],np.float32),int(rows[i]['tx_row']),int(rows[i]['tx_col'])) for i in [0,154]]
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
def score(i,pr):
 out=[]
 for b in range(2):
  keep=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool);e=pr[b][keep].astype(float)-Y[b][i][keep].astype(float);out.append(float(np.sqrt(np.mean(e*e))))
 return np.array(out)
refs={};base_times=[]
for i,h,r,c in inputs:
 t=time.perf_counter();refs[i]=net.predict(h,r,c);torch.cuda.synchronize();base_times.append(time.perf_counter()-t)
net.chunk=4096;net.precision='bf16';before=[]
for i,h,r,c in inputs:
 for j in range(2):
  t=time.perf_counter();pr=net.predict(h,r,c);torch.cuda.synchronize()
  if j==1:before.append({'index':i,'seconds':time.perf_counter()-t,'RMSE':score(i,pr).tolist()})
pointers=[net.high.fields.data_ptr(),net.high.map_info.data_ptr(),net.high.margins.data_ptr()]
for mod in [net.teacher,net.low,net.high]:mod.forward=torch.compile(mod.forward,mode='default',fullgraph=True,dynamic=False)
records=[];t0=time.time()
for i,h,r,c in inputs:
 times=[]
 for k in range(4):
  t=time.perf_counter();pr=net.predict(h,r,c);torch.cuda.synchronize();times.append(time.perf_counter()-t)
 assert pointers==[net.high.fields.data_ptr(),net.high.map_info.data_ptr(),net.high.margins.data_ptr()]
 diff=pr.astype(float)-refs[i].astype(float);delta=score(i,pr)-score(i,refs[i])
 rec={'index':i,'all_seconds':times,'warm_seconds_both_bands':float(np.mean(times[1:])),'RMS_diff_db':np.sqrt((diff*diff).mean((1,2))).tolist(),'max_abs_diff_db':float(abs(diff).max()),'RMSE':score(i,pr).tolist(),'RMSE_delta_vs_canonical':delta.tolist()}
 assert max(abs(delta))<.01,rec
 records.append(rec);print(json.dumps(rec),flush=True)
atomic_json(R/'audit/c54_band_experts_20261102/compiled_runtime.json',{'gpu':gpu,'mode':'torch.compile default fullgraph=True dynamic=False;BF16correction4096;teacherBF164096','uncompiled_canonical_seconds':base_times,'uncompiled_BF16':before,'records':records,'seconds_including_compile':time.time()-t0,'two_different_FIT_terrains':True,'fixed_buffer_addresses':True,'official_VAL_used':False,'both_maps_under_500ms':all(v['warm_seconds_both_bands']<.5 for v in records),'limit':'FIT numerical QA only; no full DEV compiled replay or official hardware test'})
