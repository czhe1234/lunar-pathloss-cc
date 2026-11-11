import os,sys,time,json,fcntl,gc
from pathlib import Path
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import *
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=gpu_guard();torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
data=ResidentData('train');fit=data.indices('fit');torch.manual_seed(20261031)
model=model_for();torch.nn.init.normal_(model.head.weight,std=.02);x,y,m=data.batch(fit[:32])
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=model(x).float()
t0=time.time();net=torch.compile(model)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=net(x).float()
torch.cuda.synchronize();difference=(reference-actual).abs();errmax=float(difference.max());errmean=float(difference.mean())
print(json.dumps({'event':'compiled_forward','seconds':time.time()-t0,'max_abs_difference_db':errmax*SCALE,'mean_abs_difference_db':errmean*SCALE}),flush=True)
assert errmean*SCALE<.1 and errmax*SCALE<1.,'Compiled forward differs excessively'
opt=torch.optim.AdamW(model.parameters(),lr=.001,fused=True);warm_start=time.time()
for step in range(5+20):
 if step==5:torch.cuda.synchronize();begin=time.perf_counter()
 ii=fit[(np.arange(32)+step*32)%len(fit)];x,y,m=data.batch(ii,'raw',step%8);opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x);loss=masked_loss(pred,y,m)
 loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,foreach=True,error_if_nonfinite=True);opt.step()
 if step==0:print(json.dumps({'event':'compiled_backward','seconds':time.time()-warm_start}),flush=True)
torch.cuda.synchronize();elapsed=time.perf_counter()-begin
result={'device':info,'batch':32,'maps_per_second':640/elapsed,'step_seconds':elapsed/20,'total_probe_seconds':time.time()-t0,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'max_abs_forward_difference_db':errmax*SCALE,'mean_abs_forward_difference_db':errmean*SCALE,'gradient_clip_included':True}
atomic_json(ROOT/'audit/compile_benchmark.json',result);print('COMPILE_RESULT '+json.dumps(result),flush=True)
