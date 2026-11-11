from c43_train import setup,Data,Model,A,gather,atomic_json
from c43_inference import Predictor,R
import torch,numpy as np,json,csv
gpu=setup(20261102);run=R/'runs/c43_path_fit64';meta=json.loads((run/'dev64_full.json').read_text())
ids=meta['indices'][:2];data=Data(ids);dat=data.batch(ids)
p=Predictor(run/'best.pt');row=data.rows[ids[0]];h=data.h[data.hmrow[data.lookup[ids[0]]]].cpu().numpy()
def from_state(net,x,tx,fs,amp):
    pr=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        st=net.encode(x,tx)
        for lo in range(0,65536,2048):
            pix=torch.arange(lo,lo+2048,device='cuda')[None].expand(len(x),-1)
            q=net.decode(st,pix).float()*20+gather(fs,pix)
            pr.append(q)
    return torch.cat(pr,1)[0].T.reshape(2,256,256).cpu().numpy()
p0=p.predict(h,int(row['tx_row']),int(row['tx_col']))
one=from_state(p.net,dat[0][:1].contiguous(memory_format=torch.channels_last),dat[1][:1],dat[2][:1],True)
two=from_state(p.net,dat[0],dat[1],dat[2],True)
saved=np.load(run/'dev64_full.npy',mmap_mode='r')[0].T.reshape(2,256,256)
one32=from_state(p.net,dat[0][:1].contiguous(memory_format=torch.channels_last),dat[1][:1],dat[2][:1],False)
two32=from_state(p.net,dat[0],dat[1],dat[2],False)
def diff(a,b):
    d=a.astype(float)-b.astype(float);return {'RMSE_db':float(np.sqrt((d*d).mean())),'max_db':float(abs(d).max())}
out={'input_only_vs_resident_single':diff(p0,one),'single_vs_batch2_BF16':diff(one,two),
     'batch2_vs_saved':diff(two,saved),'single_vs_batch2_FP32':diff(one32,two32),'BF16_vs_FP32_single':diff(one,one32)}
atomic_json(A/'interface_debug64.json',out);print(json.dumps(out),flush=True)
