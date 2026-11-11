from c43_train import setup,Data,Model,A,gather,atomic_json
from c43_inference import Predictor,R
import torch,numpy as np,json,time
setup(20261102);run=R/'runs/c43_path_fit64';meta=json.loads((run/'dev64_full.json').read_text())
js=[0,64,127,255];ids=[meta['indices'][j] for j in js];data=Data(ids)
p=Predictor(run/'best.pt',amp=False);torch.backends.cudnn.benchmark=True
records=[]
for j,i in zip(js,ids):
    row=data.rows[i];idx=data.lookup[i]
    h=data.h[data.hmrow[idx]].cpu().numpy()
    torch.cuda.synchronize();start=time.perf_counter()
    output=p.predict(h,int(row['tx_row']),int(row['tx_col']))
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    dat=data.batch([i]);pred=[]
    with torch.inference_mode():
        state=p.net.encode(dat[0],dat[1])
        for lo in range(0,65536,2048):
            pix=torch.arange(lo,lo+2048,device='cuda')[None]
            pred.append(p.net.decode(state,pix)*20+gather(dat[2],pix))
    direct=torch.cat(pred,1)[0].T.reshape(2,256,256).cpu().numpy()
    d=output.astype(float)-direct.astype(float);rms=float(np.sqrt((d*d).mean()));assert rms<1e-4
    saved=np.load(run/'dev64_full.npy',mmap_mode='r')[j].T.reshape(2,256,256)
    records.append({'index':i,'seconds':seconds,'same_single_input_interface_RMSE_db':rms,
                    'difference_from_old_BF16_batch_RMSE_db':float(np.sqrt(((output.astype(float)-saved.astype(float))**2).mean()))})
out={'precision':'FP32, TF32 disabled','gpu':p.gpu,'records':records,
     'warm_seconds_mean':float(np.mean([r['seconds'] for r in records[1:]])),
     'cause':'BF16/TF32 arithmetic can vary with batch/kernel; input-only and resident single inputs agree. Keep original failures; final precision must be explicit.'}
atomic_json(A/'precision64.json',out);print(json.dumps(out),flush=True)
