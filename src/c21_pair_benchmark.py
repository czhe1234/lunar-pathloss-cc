import json,time,numpy as np,torch
from c21_train import setup,model,gather,A,evaluate_loaded
from c21_pair import evaluate_batched,combine
from cc_core import atomic_json
gpu=setup(20261123);dat=tuple(torch.from_numpy(np.load(A/f'small_{n}.npy')).cuda().repeat(2,*([1]*(np.load(A/f'small_{n}.npy',mmap_mode='r').ndim-1))) for n in ['x','tx','fs','y','m'])
nets=[];opts=[];rows=[]
for geo in [False,True]:
 torch.manual_seed(20261123);net=model(geo);nets.append(net);opts.append(torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.0005))
x,tx,fs,y,m=dat;pix=torch.randint(65536,(8,2048),device='cuda');base=gather(fs,pix);yy=gather(y,pix);mm=gather(m,pix);times=[[],[]]
for step in range(6):
 for k,(net,opt) in enumerate(zip(nets,opts)):
  torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x,tx,pix,512)*20+base
  loss=(((p-yy).square()*mm).sum((0,1))/mm.sum((0,1))).mean()/400
  loss.backward();opt.step();torch.cuda.synchronize();times[k].append(time.time()-t)
for k,net in enumerate(nets):
 t=time.time();met,p=evaluate_batched(net,tuple(v[:4] for v in dat));torch.cuda.synchronize();sec=time.time()-t
 rows.append({'geometry':bool(k),'train_seconds':times[k],'dense4maps_seconds':sec,'evaluation':met})
atomic_json(A/'paired_benchmark.json',{'gpu':gpu,'batch':8,'queries':2048,'peak_GB':torch.cuda.max_memory_allocated()/1e9,'models':rows})
print(json.dumps({'peak_GB':torch.cuda.max_memory_allocated()/1e9,'timings':times,'dense4maps_seconds':[r['dense4maps_seconds'] for r in rows]}),flush=True)
