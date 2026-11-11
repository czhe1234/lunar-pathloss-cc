"""H/TX-only dense feature and frozen-teacher cache for four FIT terrains."""
import sys,time,csv,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c47_train import tiny_selection,load_teacher,teacher_net
from src.c47_inference import Predictor as Features
from src.c61_inference import Predictor as Frame
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.6)
sel,ids,terrains=tiny_selection();rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
d=R/'features/c62_dense4';d.mkdir(exist_ok=False)
s,c=load_teacher();teacher=teacher_net(s,c);f=object.__new__(Features);yy,xx=np.mgrid[:256,:256];f.yy=yy.astype(np.float32);f.xx=xx.astype(np.float32)
fr=object.__new__(Frame);fr.yy=f.yy;fr.xx=f.xx
out={k:np.lib.format.open_memmap(d/(k+'.npy'),mode='w+',dtype=np.float32,shape=shape) for k,shape in [('x',(4,65536,294)),('fspl',(4,65536,2)),('base',(4,65536,2)),('frame',(4,13,256,256))]}
oldx=np.load(R/'features/c38_dense/fit_x.npy',mmap_mode='r');pix=np.load(R/'features/c38_dense/fit_pixel.npy',mmap_mode='r');maxerr=0.;t=time.time()
for j,i in enumerate(ids):
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);r=int(row['tx_row']);c=int(row['tx_col']);torch.set_float32_matmul_precision('highest')
 x,fs=f.features(h,r,c);out['x'][j]=x.cpu().numpy();out['fspl'][j]=fs.cpu().numpy();out['frame'][j]=fr.frame(h,r,c)[0].cpu().numpy()
 pp=np.array(pix[sel[j*1024:(j+1)*1024]],np.int64);err=float(np.max(abs(out['x'][j,pp]-oldx[sel[j*1024:(j+1)*1024]])));maxerr=max(err,maxerr);assert err<1e-4,err
 torch.set_float32_matmul_precision('high')
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  q=torch.cat([teacher(x[lo:lo+4096]).mean(1).float() for lo in range(0,65536,4096)])
 out['base'][j]=q.cpu().numpy();print(json.dumps({'map':i,'x_maxdiff':err,'seconds':time.time()-t}),flush=True)
for v in out.values():v.flush()
atomic_json(d/'meta.json',{'complete':True,'input_only':True,'labels_masks_read':False,'map_indices':list(map(int,ids)),'terrain_ids':terrains,'teacher':'c38_1024_continue/best.pt','teacher_batch':4096,'x_maxdiff_vs_original_known_points':maxerr,'gpu':gpu,'seconds':time.time()-t})
