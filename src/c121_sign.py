"""Controlled readout of existing native descriptors, fixed identical tail subset."""
import sys,json,time,argparse,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from catboost import CatBoostClassifier,Pool
A=R/'audit/c121_pairs'
def features(x,z,mode):
 if mode=='old':return x
 v=z[:,:141].reshape(-1,3,47);meta=z[:,141:]
 if mode=='mean':return np.c_[x,(v*meta[:,:3,None]).sum(1),meta]
 return np.c_[x,z]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['old','mean','donor'],required=True);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.05);out=R/f'runs/c121_{a.mode}';out.mkdir(exist_ok=False)
 for b in range(2):
  pools={};meta={}
  for part in ['fit','monitor']:
   P=R/'audit/c101_residual'/part;ix=np.load(A/f'tail_{part}/rows.npy');t=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r');res=t[ix,b]-base[ix,b]
   keep=(t[ix,2+b]>0)&(np.abs(res)>=5);rows=ix[keep];label=(res[keep]>=5).astype('int32')
   old=np.c_[np.load(P/'common.npy',mmap_mode='r')[rows],np.load(P/'precise.npy',mmap_mode='r')[rows]]
   x=features(old,np.load(A/f'tail_{part}/donors.npy',mmap_mode='r')[keep],a.mode)
   weight=np.load(P/'weight.npy',mmap_mode='r')[rows] if part=='fit' else None
   pools[part]=Pool(x,label,weight=weight);meta[part]=dict(rows=len(rows),positive_fraction=float(label.mean()))
   np.save(out/f'{part}_rows{b}.npy',rows)
  cfg=dict(iterations=2000,depth=9,learning_rate=.05,loss_function='Logloss',eval_metric='Logloss',l2_leaf_reg=10,border_count=254,random_seed=121041,task_type='GPU',devices='0',gpu_ram_part=.5,thread_count=4,early_stopping_rounds=150,bootstrap_type='Bayesian',bagging_temperature=1.,boost_from_average=False,train_dir=str(out/f'logs{b}'))
  atomic_json(out/f'config{b}.json',dict(params=cfg,GPU=g,data=meta,inputs=x.shape[1],oracle_gate=True,official_val_used=False))
  model=CatBoostClassifier(**cfg);start=time.monotonic();model.fit(pools['fit'],eval_set=pools['monitor'],verbose=200,use_best_model=True);model.save_model(str(out/f'band{b}.cbm'))
  prob=model.predict_proba(pools['monitor'],thread_count=4)[:,1];y=pools['monitor'].get_label();acc=float(np.mean((prob>=.5)==y))
  loss=float(-np.mean(y*np.log(prob.clip(1e-9,1))+(1-y)*np.log((1-prob).clip(1e-9,1))))
  np.save(out/f'monitor_prob{b}.npy',prob.astype('float32'));np.save(out/f'importance{b}.npy',model.get_feature_importance())
  result=dict(mode=a.mode,band=b,conditional_sign_accuracy=acc,majority_accuracy=max(meta['monitor']['positive_fraction'],1-meta['monitor']['positive_fraction']),logloss=loss,trees=model.tree_count_,seconds=time.monotonic()-start,oracle_gate=True,official_val_used=False)
  atomic_json(out/f'result{b}.json',result);print('RESULT',result,flush=True)
if __name__=='__main__':main()
