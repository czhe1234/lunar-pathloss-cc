"""Fixed-data continuous residual comparison: candidate spectrum vs conserved total power."""
import sys,json,time,argparse,hashlib,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from catboost import CatBoostRegressor,Pool
A=R/'audit/c124_spectrum';S=R/'audit/c101_residual'
def array(part,name):return np.load(S/part/(name+'.npy'),mmap_mode='r')
def features(part,rows,mode):
 total=np.load(A/part/'total.npy',mmap_mode='r')[rows]
 if mode=='spectrum':
  hist=np.load(A/part/'spectrum.npy',mmap_mode='r')[rows].astype('float32')
 else:
  # Equal input count, equal sum. Both arms also retain the exact total as two explicit columns.
  uniform=np.maximum(total-np.log10(128)/4,-9.5).astype('float16').astype('float32')
  hist=np.repeat(uniform,128,axis=1)
 return np.c_[array(part,'common')[rows],array(part,'precise')[rows],total,hist].astype('float32')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['spectrum','total'],required=True);a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.03)
 out=R/f'runs/c124_{a.mode}';out.mkdir(exist_ok=False)
 for b in range(2):
  pools={};stats={};selections={}
  for split in ['fit','monitor']:
   assert (A/split/'merged.json').exists()
   target=array(split,'target');rows=np.flatnonzero(target[:,b+2]>0);selections[split]=rows
   x=features(split,rows,a.mode);y=target[rows,b]-array(split,'base')[rows,b]
   assert np.isfinite(x).all() and np.isfinite(y).all()
   weight=array(split,'weight')[rows] if split=='fit' else None
   pools[split]=Pool(x,y,weight=weight);del x;gc.collect()
   stats[split]=dict(points=len(rows),base_rmse=float(np.sqrt(np.mean(y.astype('float64')**2))))
  config=dict(iterations=3000,depth=10,learning_rate=.05,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=10,border_count=254,random_seed=124041,task_type='GPU',devices='0',gpu_ram_part=.65,thread_count=6,early_stopping_rounds=200,boosting_type='Plain',bootstrap_type='Bayesian',bagging_temperature=1.,boost_from_average=False,train_dir=str(out/f'catboost{b}'))
  atomic_json(out/f'config{b}.json',dict(params=config,GPU=g,stats=stats,mode=a.mode,inputs=427,official_val_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  model=CatBoostRegressor(**config);start=time.monotonic();model.fit(pools['fit'],eval_set=pools['monitor'],use_best_model=True,verbose=100)
  model.save_model(str(out/f'band{b}.cbm'))
  pred=model.predict(pools['monitor'],task_type='CPU',thread_count=6).astype('float32');y=pools['monitor'].get_label()
  rmse=float(np.sqrt(np.mean((pred.astype('float64')-y)**2)))
  np.save(out/f'monitor_delta{b}.npy',pred);np.save(out/f'monitor_rows{b}.npy',selections['monitor'])
  np.save(out/f'importance{b}.npy',model.get_feature_importance())
  result=dict(status='complete',mode=a.mode,band=b,monitor_rmse=rmse,C82=stats['monitor']['base_rmse'],gain=stats['monitor']['base_rmse']-rmse,trees=model.tree_count_,best_iteration=model.get_best_iteration(),seconds=time.monotonic()-start,GPU=g,official_val_used=False)
  atomic_json(out/f'result{b}.json',result);print('RESULT',result,flush=True)
  del pools,model;gc.collect()
 atomic_json(out/'complete.json',dict(status='complete',mode=a.mode,official_val_used=False))
if __name__=='__main__':main()
