import sys,json,time,fcntl,csv,gc,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,gpu_guard,atomic_json
from c4_geometry import make_inputs
from c5_model import Model,mix_losses_db
import numpy as np,torch
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();torch.set_num_threads(8);torch.manual_seed(20261101);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
model=Model().cuda().to(memory_format=torch.channels_last)
v=torch.tensor([0.,.5,1.],device='cuda');d=torch.zeros_like(v);s=torch.ones_like(v)*20
p=mix_losses_db(d,s,v,torch.ones_like(v));ref=-10*torch.log10(v+(1-v)*.01);assert torch.allclose(p,ref,atol=1e-5)
p2=mix_losses_db(d,s,v,torch.zeros_like(v));assert torch.allclose(p2,(1-v)*20,atol=1e-5)
x=torch.zeros(1,31,256,256,device='cuda');xx=torch.arange(256,device='cuda')[None,None,:].expand(1,256,256)
e=.02*(xx-128);x[:,11:19]=torch.asinh(e[:,None]/2)/3;x[:,19]=torch.asinh((e+.3)/2)/3;q=torch.zeros(1,9,256,256,device='cuda',requires_grad=True)
v=model.observation_visibility(x,q);assert float((v[0,0,:,128]-.5).abs().max())<1e-5
v.mean().backward();assert torch.isfinite(q.grad).all() and q.grad[:,0].abs().sum()>0
checks={'passed':True,'device':info,'power_domain_mixture_reference':p.tolist(),'dB_domain_mixture_reference':p2.tolist(),'symmetric_boundary_visibility':float(v[0,0,128,128]),'visibility_calibration_gradient_finite_and_nonzero':True}
atomic_json(ROOT/'audit/c5_checks.json',checks);print(json.dumps(checks),flush=True)
del model,x,q,v,p,p2,e,d,s;gc.collect();torch.cuda.empty_cache()
rows=list(csv.DictReader((DATA/'train_index.csv').open()));dev=set(json.loads((ROOT/'audit/internal_split.json').read_text())['dev_terrain_ids']);ids=[i for i,r in enumerate(rows) if r['terrain_id'] not in dev][:64]
# Reserve the same storage as the complete training geometry/height cache.
reserve_g=torch.empty((len(rows),20,256,256),device='cuda',dtype=torch.float16)
reserve_h=torch.empty((6120,256,256),device='cuda',dtype=torch.float32)
ck=torch.load(ROOT/'runs/c4_critical_length_fast_s20261031/best.pt',map_location='cpu',weights_only=False)
hm=np.load(DATA/'train_hm.npy',mmap_mode='r');g1=np.load(ROOT/'features/c3/train.npy',mmap_mode='r');g2=np.load(ROOT/'features/c4_extra/train.npy',mmap_mode='r')
records=[]
for batch in ([32] if '--compiled' in sys.argv else [16,32,48,64]):
 model=None;opt=None;x=None;fs=None;y=None;m=None;geo=None;h=None;loss=None;pr=None
 try:
  ii=ids[:batch];rr=[rows[i] for i in ii];h=torch.from_numpy(np.array(hm[[int(r['hm_row']) for r in rr]],copy=True)).cuda();r=torch.tensor([int(z['tx_row']) for z in rr],device='cuda');c=torch.tensor([int(z['tx_col']) for z in rr],device='cuda')
  geo=torch.from_numpy(np.concatenate([g1[ii],g2[ii]],axis=1)).cuda();x,fs=make_inputs(h,r,c,geo);del geo,h;geo=h=None
  y=torch.from_numpy(np.stack([np.load(DATA/f'train_pl{b}.npy',mmap_mode='r')[ii] for b in ['415','58']],1).astype(np.float32)).cuda()
  m=torch.from_numpy(np.stack([np.unpackbits(np.load(DATA/f'train_mask{b}.npy',mmap_mode='r')[ii],axis=1,bitorder='big').reshape(batch,256,256) for b in ['415','58']],1).astype(np.float32)).cuda()
  model=Model().cuda().to(memory_format=torch.channels_last);missing=model.load_state_dict(ck['model'],strict=False);assert all(k.startswith('observation.') for k in missing.missing_keys)
  opt=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
  net=torch.compile(model) if '--compiled' in sys.argv else model
  def step():
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pr=net(x)+fs
   loss=((((pr-y)/20).square()*m).sum((0,2,3))/m.sum((0,2,3))).mean();assert torch.isfinite(loss);loss.backward();opt.step();return float(loss.detach())
  for _ in range(2):value=step()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  for _ in range(4):value=step()
  torch.cuda.synchronize();sec=time.perf_counter()-t
  row={'batch':batch,'mode':('compiled' if '--compiled' in sys.argv else 'eager'),'steps':4,'seconds':sec,'maps_per_second':4*batch/sec,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'last_discarded_fit_loss':value};records.append(row);print(json.dumps(row),flush=True)
 except torch.OutOfMemoryError as e:
  records.append({'batch':batch,'mode':('compiled' if '--compiled' in sys.argv else 'eager'),'out_of_memory':True});print(json.dumps(records[-1]),flush=True)
 finally:
  del model,opt,x,fs,y,m,geo,h,loss,pr;gc.collect();torch.cuda.empty_cache()
atomic_json(ROOT/('audit/c5_benchmark_compiled.json' if '--compiled' in sys.argv else 'audit/c5_benchmark.json'),{'device':info,'fit_only_discarded_updates':True,'cache_storage_reserved_GB':(reserve_g.numel()*2+reserve_h.numel()*4)/1e9,'records':records})
