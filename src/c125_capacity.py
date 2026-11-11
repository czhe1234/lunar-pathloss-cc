"""Training-only model selection: can a high-capacity learner fit all frozen tail points?"""
import sys,json,time,gc,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from catboost import CatBoostRegressor,Pool
A=R/'audit/c125_tail';S=R/'audit/c101_residual';T=R/'audit/c120_sign'
g=gpu_guard();torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.03);out=R/'runs/c125_capacity_old';out.mkdir(exist_ok=False)
for b in range(2):
 pools={};yy={};ww={}
 for part in ['fit','monitor']:
  P=S/part;t=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r');u=np.load(T/(part+'_rows.npy'));r=t[u,b]-base[u,b];keep=(t[u,b+2]>0)&(abs(r)>=5);ix=u[keep]
  x=np.asfortranarray(np.c_[np.load(P/'common.npy',mmap_mode='r')[ix],np.load(P/'precise.npy',mmap_mode='r')[ix]],dtype='float32')
  y=r[keep];w=np.load(P/'weight.npy',mmap_mode='r')[ix] if part=='fit' else np.ones(len(ix),np.float32)
  pools[part]=Pool(x,y,weight=w if part=='fit' else None);yy[part]=y;ww[part]=w
 cfg=dict(iterations=3000,depth=12,learning_rate=.15,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=1,border_count=254,random_seed=125241,task_type='GPU',devices='0',gpu_ram_part=.65,thread_count=6,boosting_type='Plain',bootstrap_type='No',boost_from_average=True,train_dir=str(out/f'logs{b}'))
 atomic_json(out/f'config{b}.json',dict(params=cfg,GPU=g,scope='capacity diagnosis only, fixed full budget, no monitor checkpoint selection',input=169,official_val_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 model=CatBoostRegressor(**cfg);st=time.monotonic();model.fit(pools['fit'],eval_set=pools['monitor'],use_best_model=False,verbose=500);model.save_model(str(out/f'band{b}.cbm'))
 result=dict(band=b,trees=model.tree_count_,capacity_diagnostic=True,selected_on_monitor=False,official_val_used=False)
 for part in ['fit','monitor']:
  d=model.predict(pools[part],thread_count=6).astype('float64');y=yy[part].astype('float64');w=ww[part].astype('float64')
  result[part]=dict(tail_RMSE=float(np.sqrt(np.mean((d-y)**2))),weighted_tail_RMSE=float(np.sqrt(np.sum(w*(d-y)**2)/w.sum())),sign_accuracy=float(np.mean((d>=0)==(y>=0))),points=len(y))
  np.save(out/f'{part}_delta{b}.npy',d.astype('float32'))
 result['seconds']=time.monotonic()-st;atomic_json(out/f'result{b}.json',result);atomic_json(out/f'curves{b}.json',model.get_evals_result());print('RESULT',result,flush=True);del pools,model;gc.collect()
atomic_json(out/'complete.json',dict(status='complete',scope='capacity only'))
