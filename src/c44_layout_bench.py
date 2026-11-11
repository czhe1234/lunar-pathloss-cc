from c44_train import setup,Data,A,gather,atomic_json
from c44_model import Model as Old
from c44_model_2d import Model as New
import torch,numpy as np,json,time
gpu=setup(20261102);p=json.loads((A/'protocol.json').read_text());data=Data(p['fit64_ids'][:16]);dat=data.batch(p['fit64_ids'][:16])
torch.manual_seed(20261102);old=Old('path').cuda().to(memory_format=torch.channels_last)
torch.manual_seed(20261102);new=New('path').cuda().to(memory_format=torch.channels_last)
a0=old.state_dict();a1=new.state_dict()
assert all(torch.equal(v.reshape(-1),a1[k].reshape(-1)) for k,v in a0.items())
pix=torch.arange(0,65536,1024,device='cuda')[None].expand(2,-1)
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
with torch.inference_mode():
 one=old(dat[0][:2].contiguous(memory_format=torch.channels_last),dat[1][:2],pix)
 two=new(dat[0][:2].contiguous(memory_format=torch.channels_last),dat[1][:2],pix)
difference=float((one-two).abs().max());assert difference<1e-4,difference
del old,new;torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
records=[]
for cls,label in [(Old,'conv1d'),(New,'equivalent_1xL_channels_last')]:
 torch.manual_seed(20261102);net=cls('path').cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0002)
 pix=torch.randint(65536,(16,2048),device='cuda');times=[]
 for k in range(6):
  torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):q=net(dat[0],dat[1],pix)
  y=(gather(dat[3],pix).float()-gather(dat[2],pix))/20;m=gather(dat[4],pix).float()
  loss=(((q.float()-y).square()*m).sum((0,1))/m.sum((0,1))).mean();loss.backward();opt.step()
  torch.cuda.synchronize();times.append(time.perf_counter()-start)
 records.append({'layout':label,'step_seconds':float(np.median(times[2:]))})
 print(json.dumps(records[-1]),flush=True)
 del opt,net;torch.cuda.empty_cache()
atomic_json(A/'layout_benchmark.json',{'gpu':gpu,'FP32_max_output_difference':difference,'flattened_initial_weights_identical':True,'benchmarks':records})
