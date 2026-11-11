from c43_train import setup,Data,Model,gather,A,atomic_json
import torch,numpy as np,json,time
gpu=setup(20261102);p=json.loads((A/'protocol.json').read_text());data=Data(p['fit64_ids'][:32])
net=Model('path').cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=.0002)
dat=data.batch(p['fit64_ids'][:32]);pix=torch.randint(65536,(32,2048),device='cuda');times=[]
for k in range(6):
    torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):q=net(dat[0],dat[1],pix)
    y=(gather(dat[3],pix).float()-gather(dat[2],pix))/20;m=gather(dat[4],pix).float()
    loss=(((q.float()-y).square()*m).sum((0,1))/m.sum((0,1))).mean()
    loss.backward();opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
s=float(np.median(times[2:]));r={'batch':32,'step_seconds':s,'maps_per_second':32/s,'peak_GB':torch.cuda.max_memory_allocated()/1e9,'gpu':gpu}
atomic_json(A/'benchmark32.json',r);print(json.dumps(r),flush=True)
