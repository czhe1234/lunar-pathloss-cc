from c44_train import setup,Data,Model,A,atomic_json,gather
import torch,numpy as np,json,time
gpu=setup(20261102);p=json.loads((A/'protocol.json').read_text());data=Data(p['fit64_ids'][:1]);dat=data.batch(p['fit64_ids'][:1])
net=Model('path').cuda().to(memory_format=torch.channels_last);net.eval()
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
rows=[];reference=None
with torch.inference_mode():
 for chunk in [2048,8192]:
  times=[];pr=None
  for k in range(3):
   torch.cuda.synchronize();start=time.perf_counter();state=net.encode(dat[0],dat[1]);parts=[]
   for lo in range(0,65536,chunk):
    pix=torch.arange(lo,min(lo+chunk,65536),device='cuda')[None]
    parts.append(net.decode(state,pix)*20+gather(dat[2],pix))
   pr=torch.cat(parts,1);torch.cuda.synchronize();times.append(time.perf_counter()-start)
  if reference is None:reference=pr.clone()
  diff=float((pr-reference).abs().max());assert diff<1e-3
  rows.append({'chunk':chunk,'warm_seconds':float(np.mean(times[1:])),'max_output_difference_db':diff})
  print(json.dumps(rows[-1]),flush=True)
atomic_json(A/'inference_benchmark.json',{'gpu':gpu,'precision':'FP32 noTF32','rows':rows,'scope':'network and FSPL, raw descriptors already on GPU; input preprocessing adds unbenchmarked small overhead'})
