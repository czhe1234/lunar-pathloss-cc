"""Evaluate frozen C38 on seen pixels and unseen pixels of the same FIT maps."""
import sys,csv,json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c47_inference import Predictor as Features
from src.c47_train import load_teacher,teacher_net
import numpy as np,torch
gpu=gpu_guard();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.4)
s,c=load_teacher();net=teacher_net(s,c);f=object.__new__(Features);yy,xx=np.mgrid[:256,:256];f.yy=yy.astype(np.float32);f.xx=xx.astype(np.float32)
ids=json.loads((R/'audit/c43_architecture_20261102/protocol.json').read_text())['fit64_ids']
maps=np.load(R/'features/c38_dense/fit_map_indices.npy');lookup={int(i):j for j,i in enumerate(maps)}
pixels=np.load(R/'features/c38_dense/fit_pixel.npy',mmap_mode='r')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
acc={k:[np.zeros(2),np.zeros(2)] for k in ['seen1024','unseen64512','all']};rec=[];t=time.time()
for j,i in enumerate(ids):
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32)
 torch.set_float32_matmul_precision('highest');x,fs=f.features(h,int(row['tx_row']),int(row['tx_col']))
 torch.set_float32_matmul_precision('high');out=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for lo in range(0,65536,4096):out.append(net(x[lo:lo+4096]).mean(1).float()*20+fs[lo:lo+4096])
 pr=torch.cat(out).cpu().numpy()
 seen=np.zeros(65536,dtype=bool);pos=lookup[i];seen[np.array(pixels[pos*1024:(pos+1)*1024],np.int64)]=True;assert seen.sum()==1024
 rr={'index':i,'terrain':row['terrain_id']}
 for name,sel in [('seen1024',seen),('unseen64512',~seen),('all',np.ones(65536,dtype=bool))]:
  ss=[];cc=[]
  for b in range(2):
   keep=np.unpackbits(M[b][i],bitorder='big').astype(bool)&sel;e=pr[keep,b].astype(float)-Y[b][i].ravel()[keep].astype(float)
   ss.append(float(e@e));cc.append(int(keep.sum()))
  acc[name][0]+=ss;acc[name][1]+=cc;rr[name]={'SSE':ss,'count':cc}
 rec.append(rr)
 if j%40==0:print(json.dumps({'maps':j+1,'total':len(ids),'seconds':time.time()-t}),flush=True)
out={'maps':len(ids),'terrains':64,'groups':{k:{'RMSE':np.sqrt(v[0]/v[1]).tolist(),'SSE':v[0].tolist(),'count':v[1].tolist()} for k,v in acc.items()},'gpu':gpu,'seconds':time.time()-t,'scope':'Same 64 FIT terrains, all264maps; no official VAL; frozen C38 model; masks only scorer','permap':rec}
atomic_json(R/'audit/c59_capacity_coverage_20261102/coverage.json',out);print(json.dumps({k:v for k,v in out.items() if k!='permap'}),flush=True)
