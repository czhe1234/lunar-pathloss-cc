import sys,os,fcntl,time,json,gc,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import *
lock=(ROOT/'.gpu3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
check_source_unchanged();info=gpu_guard();print(json.dumps(info),flush=True)
random.seed(20261031);np.random.seed(20261031);torch.manual_seed(20261031)
data=ResidentData('train');fit=data.indices('fit');dev=data.indices('dev')
assert not set(fit)&set(dev)
# Validate geometry conventions after all eight coordinate transforms.
base=[0,100,1000,10000]
for k in range(8):
 x,y,m=data.batch(base,'geometry',k)
 actual=x[:,1].flatten(1).argmax(1);rr,cc=d4_coords(data.txr[base],data.txc[base],k)
 assert torch.equal(actual,rr*256+cc)
 assert torch.allclose(x[:,0]*100,d4(data.h[data.hmrow[base]],k),atol=5e-5)
 assert torch.equal(y,d4(data.y[base].float(),k))
 assert torch.equal(m,d4(data.mask[base].float(),k))
 assert torch.isfinite(x).all()
# Independent scalar calculation: band MSE 4 and 16 dB^2 -> equal-weight loss 10/1600.
p=torch.zeros((1,2,2,2),device='cuda');t=torch.tensor([[[[82.,84.],[90.,92.]],[[84.,88.],[90.,92.]]]],device='cuda');m=torch.tensor([[[[1.,0.],[0.,0.]],[[1.,0.],[0.,0.]]]],device='cuda')
assert abs(float(masked_loss(p,t,m))*SCALE*SCALE-10)<1e-5
print('coordinate_transform_and_loss_checks_passed',flush=True)
# Ensure the independent model can fit a few real training examples.
model=model_for();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,fused=True)
x,y,m=data.batch(fit[:4]);losses=[];t0=time.time()
for step in range(800):
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x);loss=masked_loss(pred,y,m)
 loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,foreach=True,error_if_nonfinite=True);opt.step()
 losses.append(float(loss.detach()))
 if step%100==0:print(json.dumps({'sanity_step':step,'mse_db':losses[-1]*SCALE*SCALE}),flush=True)
atomic_json(ROOT/'audit/sanity_loss_trajectory.json',{'loss_mse_db':[v*SCALE*SCALE for v in losses]})
assert np.isfinite(losses).all() and np.mean(losses[-10:])<.3*np.mean(losses[:10]),'Four-example fitting failed'
san={'initial_mean_mse_db':float(np.mean(losses[:10])*SCALE*SCALE),'final_mean_mse_db':float(np.mean(losses[-10:])*SCALE*SCALE),'steps':len(losses),'seconds':time.time()-t0}
atomic_json(ROOT/'audit/training_sanity.json',san)
print(json.dumps({'sanity':san}),flush=True)
del model,opt,x,y,m,p,t;gc.collect();torch.cuda.empty_cache()
results=[]
for bs in [16,32,64,128,256]:
 torch.manual_seed(20261031);net=model_for();opt=torch.optim.AdamW(net.parameters(),lr=.001,fused=True)
 try:
  torch.cuda.reset_peak_memory_stats();t0=time.time()
  for step in range(5+20):
   if step==5:torch.cuda.synchronize();begin=time.perf_counter()
   idx=fit[(np.arange(bs)+step*bs)%len(fit)]
   x,y,m=data.batch(idx,'raw',step%8);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x);loss=masked_loss(pred,y,m)
   loss.backward();opt.step()
  torch.cuda.synchronize();elapsed=time.perf_counter()-begin
  result={'batch':bs,'maps_per_second':bs*20/elapsed,'step_seconds':elapsed/20,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,'peak_reserved_GB':torch.cuda.max_memory_reserved()/1e9,'finite_loss':bool(torch.isfinite(loss))}
 except torch.OutOfMemoryError:
  result={'batch':bs,'oom':True}
 results.append(result);print(json.dumps({'benchmark':result}),flush=True)
 del net,opt
 for name in ['x','y','m','pred','loss']:
  if name in globals():del globals()[name]
 gc.collect();torch.cuda.empty_cache()
 if result.get('oom'):break
atomic_json(ROOT/'audit/throughput_benchmark.json',{'device':info,'precision':'BF16 convolutions, FP32 output/loss; channels_last; data resident','warmup_steps':5,'timed_steps':20,'results':results})
print('BENCHMARK_COMPLETE',flush=True)
