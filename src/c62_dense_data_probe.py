"""Frozen C61 tiny model: check whether small-set fit generalizes to unseen pixels."""
import sys,csv,time,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from src.c61_inference import Predictor
from src.c47_train import tiny_selection
gpu=gpu_guard();net=Predictor(R/'runs/c61_path_tiny',precision='bf16')
selection,ids,terrains=tiny_selection();pixels=np.load(R/'features/c38_dense/fit_pixel.npy',mmap_mode='r')
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()));H=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
tiny=np.load(R/'runs/c61_path_tiny/dev_prediction.npy')
acc={k:{m:[np.zeros(2),np.zeros(2)] for m in ['C38','C61']} for k in ['seen1024','unseen64512','all']};rec=[]
for j,i in enumerate(ids):
 row=rows[i];h=np.array(H[int(row['hm_row'])],np.float32);r=int(row['tx_row']);c=int(row['tx_col'])
 t=time.time();pr=net.predict(h,r,c)
 x,fs=net.features(h,r,c);base=[];torch.set_float32_matmul_precision('high')
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for lo in range(0,65536,4096):base.append(net.net.backbone.eval()(x[lo:lo+4096]).mean(1).float()*20+fs[lo:lo+4096])
 base=torch.cat(base).cpu().numpy().T.reshape(2,256,256)
 pp=np.array(pixels[selection[j*1024:(j+1)*1024]],np.int64);seen=np.zeros(65536,bool);seen[pp]=True
 dif=pr.reshape(2,-1)[:,pp].T.astype(float)-tiny[j*1024:(j+1)*1024].astype(float)
 rowout={'index':i,'known_pixel_replay_RMS_db':np.sqrt((dif*dif).mean(0)).tolist(),'seconds_both_predictions':time.time()-t}
 for name,sel in [('seen1024',seen),('unseen64512',~seen),('all',np.ones(65536,bool))]:
  rowout[name]={}
  for label,pred in [('C38',base),('C61',pr)]:
   ss=[];cc=[]
   for b in range(2):
    keep=np.unpackbits(M[b][i],bitorder='big').astype(bool)&sel;e=pred[b].ravel()[keep].astype(float)-Y[b][i].ravel()[keep].astype(float);ss.append(float(e@e));cc.append(int(keep.sum()))
   acc[name][label][0]+=ss;acc[name][label][1]+=cc;rowout[name][label]={'RMSE':np.sqrt(np.array(ss)/cc).tolist(),'SSE':ss,'count':cc}
 rec.append(rowout);print(json.dumps(rowout),flush=True)
out={'scope':'Only the same four FIT terrains, all pixels; no official VAL; frozen tiny checkpoint','maps':list(map(int,ids)),'gpu':gpu,'groups':{k:{m:{'RMSE':np.sqrt(v[0]/v[1]).tolist(),'SSE':v[0].tolist(),'count':v[1].tolist()} for m,v in d.items()} for k,d in acc.items()},'permap':rec,'limit':'Unseen pixels on known terrains are not unseen-terrain validation'}
atomic_json(R/'audit/c61_global_terrain_20261102/tiny_unseen_pixels.json',out);print(json.dumps({k:v for k,v in out.items() if k!='permap'}),flush=True)
