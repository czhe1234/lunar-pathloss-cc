import os,sys,json,time,csv,hashlib,shutil
from pathlib import Path
os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cc_core import ROOT,DATA,atomic_json
from c3_model import Model
from c3_geometry import make_inputs
import torch,numpy as np
torch.set_num_threads(8)
run=ROOT/'runs/c3_surface_profile_s20261031';out=ROOT/'audit/c3_early_cpu_diagnostic';out.mkdir(exist_ok=False)
shutil.copy2(run/'best.pt',out/'checkpoint.pt');ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);model=Model(32);model.load_state_dict(ck['model']);model.eval();epoch=ck['epoch'];del ck
rows=list(csv.DictReader((DATA/'train_index.csv').open()));mapping=json.loads((ROOT/'runs/b0_raw_s20261031/predictions/train_dev_bf16/indices.json').read_text());dev=np.array(mapping['index_rows']);sample=np.random.default_rng(20261031).choice(len(dev),128,replace=False);ids=dev[sample]
hm=np.load(DATA/'train_hm.npy',mmap_mode='r');geo=np.load(ROOT/'features/c3/train.npy',mmap_mode='r');ys=[np.load(DATA/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']];ms=[np.load(DATA/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']];bs=[np.load(ROOT/f'runs/b0_raw_s20261031/predictions/train_dev_bf16/pl{b}.npy',mmap_mode='r') for b in ['415','58']]
states=['all','strong_clear','height_sensitive','deep_blocked'];stats={s:{'count':np.zeros(2),'c3_sse':np.zeros(2),'b0_sse':np.zeros(2),'c3_bias_sum':np.zeros(2)} for s in states};t0=time.time()
with torch.inference_mode():
 for j,i in enumerate(ids):
  row=rows[i];h=torch.from_numpy(np.array(hm[int(row['hm_row'])],copy=True))[None];r=torch.tensor([int(row['tx_row'])]);c=torch.tensor([int(row['tx_col'])]);gg=torch.from_numpy(np.array(geo[i],copy=True))[None].float();x,fs=make_inputs(h,r,c,gg);pr=(model(x)+fs)[0].numpy().astype(np.float64)
  e3=(2*np.sinh(gg[0,:8].numpy()*3)).max(0);e2=2*np.sinh(gg[0,8].numpy()*3)
  masks=[np.ones((256,256),bool),e2<-.5,(e3<0)&(e2>=0),e3>2]
  for k in range(2):
   truth=ys[k][i].astype(np.float64);valid=np.unpackbits(ms[k][i],bitorder='big').reshape(256,256).astype(bool);err=pr[k]-truth;base=bs[k][sample[j]].astype(np.float64)-truth
   for name,sel in zip(states,masks):
    use=valid&sel;s=stats[name];s['count'][k]+=use.sum();s['c3_sse'][k]+=(err[use]**2).sum();s['b0_sse'][k]+=(base[use]**2).sum();s['c3_bias_sum'][k]+=err[use].sum()
  if (j+1)%16==0:print(json.dumps({'done':j+1,'total':len(ids),'epoch':epoch,'seconds':time.time()-t0}),flush=True)
for s in stats.values():
 s['c3_rmse']=np.sqrt(s['c3_sse']/s['count']);s['b0_rmse']=np.sqrt(s['b0_sse']/s['count']);s['c3_bias']=s['c3_bias_sum']/s['count']
 for k,v in s.items():s[k]=v.tolist()
result={'checkpoint_epoch':epoch,'checkpoint_sha256':hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),'precision':'CPU FP32 diagnostic; not official or full development score','samples':len(ids),'index_rows':ids.tolist(),'states':stats,'seconds':time.time()-t0,'GPU_used':False};atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
