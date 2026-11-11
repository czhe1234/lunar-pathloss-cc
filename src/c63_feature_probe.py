"""Arbitrary-pixel pipeline parity and throughput before any large training."""
import sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c63_point_features import FeatureBank
from src.c47_train import tiny_selection,load_teacher,teacher_net
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.88);torch.set_float32_matmul_precision('highest')
t=time.time();bank=FeatureBank();print(json.dumps({'event':'feature_bank','seconds':time.time()-t,'GPU_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
sel,ids,terrains=tiny_selection();pix=np.load(R/'features/c38_dense/fit_pixel.npy',mmap_mode='r');old=np.load(R/'features/c38_dense/fit_x.npy',mmap_mode='r');oldfs=np.load(R/'features/c38_dense/fit_fspl.npy',mmap_mode='r')
maps=torch.tensor(ids,device='cuda');pp=torch.tensor(np.array(pix[sel],np.int64).reshape(4,1024),device='cuda');im=bank.image(maps);x,fs=bank.points(maps,pp,im)
ref=torch.tensor(np.array(old[sel]),device='cuda').reshape(4,1024,294);delta=(x-ref).abs();channels=delta.amax((0,1)).cpu().tolist()
rec={'maximum_feature_difference':float(delta.max()),'maxdiff_by_group':{'basic':max(channels[:11]),'geometry':max(channels[11:31]),'TX':max(channels[31:104]),'profile':max(channels[104:232]),'surface':max(channels[232:272]),'phase8':max(channels[272:280]),'phase14':max(channels[280:])},'fspl_maxdiff':float((fs-torch.tensor(np.array(oldfs[sel]),device='cuda').reshape(4,1024,2)).abs().max()),'largest_channels':sorted(enumerate(channels),key=lambda a:-a[1])[:12]}
print(json.dumps(rec),flush=True);s,c=load_teacher();teacher=teacher_net(s,c);torch.set_float32_matmul_precision('high')
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 a=teacher(x.reshape(-1,294)).mean(1).float();b=teacher(ref.reshape(-1,294)).mean(1).float()
d=(a-b)*20;rec['teacher_RMS_difference_db']=d.square().mean(0).sqrt().cpu().tolist();rec['teacher_max_difference_db']=float(d.abs().max())
times=[];maps=maps.repeat(4);pp=torch.randint(65536,(16,256),device='cuda')
for k in range(5):
 torch.cuda.synchronize();t=time.perf_counter();im=bank.image(maps);xx,ff=bank.points(maps,pp,im);torch.cuda.synchronize();times.append(time.perf_counter()-t)
rec['image_and4096point_seconds']=times;rec['warm_median_seconds']=float(np.median(times[1:]));rec['gpu']=gpu;rec['labels_masks_read']=False
atomic_json(R/'audit/c62_dense_coverage_20261102/arbitrary_feature_probe.json',rec)
assert max(rec['teacher_RMS_difference_db'])<.02,rec
print(json.dumps(rec),flush=True)
