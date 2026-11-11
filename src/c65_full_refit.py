"""Fixed-budget full-TRAIN warm refit of the internally confirmed band experts."""
import sys,time,json,math,hashlib,shutil,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json,check_source_unchanged
import numpy as np,torch
from src.c47_model import Model as Low
from src.c49_model import Model as High
from src.c49_train import terrain_fields
from src.c47_train import device_array
p=argparse.ArgumentParser();p.add_argument('--band',type=int,choices=[0,1],required=True);p.add_argument('--name',required=True);a=p.parse_args()
gpu=gpu_guard();check_source_unchanged();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.85);torch.set_float32_matmul_precision('high');torch.manual_seed(20261211);np.random.seed(20261211)
base_meta=json.loads((R/'features/c65_full_teacher/meta.json').read_text());assert base_meta['complete'] and not base_meta['labels_masks_read']
parts=[('fit','c38_dense'),('dev','c38_dev_full')];arrays={}
for key in ['x','y','fspl','mask','base']:
 arrs=[np.load(R/(f'features/c65_full_teacher/{part}_base.npy' if key=='base' else f'features/{folder}/{part}_{key}.npy'),mmap_mode='r') for part,folder in parts]
 shape=(sum(len(v) for v in arrs),)+arrs[0].shape[1:];tensor=torch.empty(shape,device='cuda',dtype=torch.float32);pos=0
 for ar in arrs:
  for lo in range(0,len(ar),262144):tensor[pos+lo:pos+min(lo+262144,len(ar))].copy_(torch.from_numpy(np.array(ar[lo:lo+262144],copy=True)))
  pos+=len(ar)
 arrays[key]=tensor
 assert pos==shape[0]
maps=np.concatenate([np.load(R/f'features/{folder}/{part}_map_indices.npy') for part,folder in parts]);assert len(maps)==25720 and len(set(maps.tolist()))==25720
arrays['map']=torch.repeat_interleave(torch.tensor(maps,device='cuda',dtype=torch.long),1024)
target=(arrays['y']-arrays['fspl'])/20-arrays['base'];assert len(target)==25720*1024
teacher=R/'runs/c38_full_refit/best.pt';ts=torch.load(teacher,map_location='cpu',weights_only=False)['model']
if a.band==0:net=Low('attention',ts).cuda();parent=R/'runs/c54_low_attention/best.pt';batch=4096
else:
 fields,info=terrain_fields();margins=device_array(R/'features/c49_visibility/train_margins.npy')
 net=High('transport',ts,fields,info,margins).cuda();parent=R/'runs/c54_high_transport/best.pt';batch=2048
parentck=torch.load(parent,map_location='cpu',weights_only=False);net.load_state_dict(parentck['model'])
assert all(torch.equal(v.detach().cpu(),parentck['model'][k]) for k,v in net.state_dict().items())
steps=round(16000*25720/23158);run=R/'runs'/a.name;run.mkdir(exist_ok=False)
for name in ['c65_full_refit.py','c47_model.py','c49_model.py','c48_model.py','c34_model.py']:shutil.copy2(R/'src'/name,run/name)
cfg={**vars(a),'kind':'attention' if a.band==0 else 'transport','gpu':gpu,'steps':steps,'seed':20261211,'batch':batch,'lr':.0003,'weight_decay':1e-4,'allTRAIN_maps':25720,'allTRAIN_terrains':6120,'points_per_map':1024,'rows':len(target),'selection':'Fixed last update count, scaled from16000 by25720/23158; no early stopping or officialVAL selection','procedure':'Warm refit from frozen C54 expert using fullTRAIN C38 baseline; original internalDEV included as TRAIN','official_VAL_used':False,'target':'(PL - fullTRAIN_C38prediction)/20','intended_band_only':a.band,'parent_checkpoint':str(parent.relative_to(R)),'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'teacher_sha256':hashlib.sha256(teacher.read_bytes()).hexdigest(),'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.py')}}
atomic_json(run/'config.json',cfg)
probe=torch.from_numpy(np.linspace(0,len(target)-1,65536,dtype=np.int64)).cuda()
@torch.inference_mode()
def evaluate():
 net.eval();ss=torch.zeros(2,device='cuda',dtype=torch.float64);cnt=torch.zeros_like(ss)
 for lo in range(0,len(probe),batch):
  j=probe[lo:lo+batch]
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(arrays['x'][j]) if a.band==0 else net(arrays['x'][j],arrays['map'][j])
  out=(q.float()+arrays['base'][j])*20+arrays['fspl'][j];e=out.double()-arrays['y'][j].double();m=arrays['mask'][j];ss+=(e.square()*m).sum(0);cnt+=m.sum(0).double()
 net.train();return {'RMSE':(ss/cnt).sqrt().cpu().tolist(),'scope':'TRAIN probe only; unused other band is not part of combined predictor'}
initial=evaluate();hist=[{'step':0,'TRAIN_probe':initial,'seconds':0}];atomic_json(run/'history.json',hist)
gen=torch.Generator(device='cuda').manual_seed(20261211);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=1e-4);t0=time.time();times=[]
for step in range(1,steps+1):
 t=time.perf_counter();j=torch.randint(len(target),(batch,),device='cuda',generator=gen);lr=.0003*min(1,step/200)*(.05+.95*.5*(1+math.cos(math.pi*(step-1)/steps)))
 for group in opt.param_groups:group['lr']=lr
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):q=net(arrays['x'][j]) if a.band==0 else net(arrays['x'][j],arrays['map'][j])
 m=arrays['mask'][j,a.band];loss=((q[:,a.band].float()-target[j,a.band]).square()*m).sum()/m.sum().clamp_min(1)
 assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(gn);opt.step()
 if step<=30:
  torch.cuda.synchronize();times.append(time.perf_counter()-t)
 if step==30:print(json.dumps({'event':'throughput','seconds_per_step':float(np.median(times[10:])),'estimated_minutes':float(np.median(times[10:])*steps/60),'GPU_GB':torch.cuda.memory_allocated()/1e9}),flush=True)
 if step%1000==0 or step==steps:
  rec={'step':step,'TRAIN_probe':evaluate(),'lr':lr,'seconds':time.time()-t0};hist.append(rec);atomic_json(run/'history.json',hist);atomic_json(run/'progress.json',rec);print(json.dumps(rec),flush=True)
torch.save({'model':net.state_dict(),'step':steps,'config':cfg},run/'best.pt')
out={'status':'complete','band':a.band,'steps':steps,'checkpoint_selection':'fixed last, not a validation-selected best','initial_TRAIN_probe':initial,'final_TRAIN_probe':evaluate(),'seconds':time.time()-t0,'gpu_peak_GB':torch.cuda.max_memory_allocated()/1e9,'official_VAL_used':False}
atomic_json(run/'result.json',out);print(json.dumps(out),flush=True)
