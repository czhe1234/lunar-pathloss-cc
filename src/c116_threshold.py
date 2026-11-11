import sys,json,time,hashlib,argparse,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from catboost import CatBoostRegressor,Pool
A=R/'audit/c116_targeted';SRC=R/'audit/c101_residual'
def load(part,name):return np.load(SRC/part/(name+'.npy'),mmap_mode='r')
def features(part,sel):return np.concatenate([load(part,'common')[sel],load(part,'precise')[sel]],1).astype(np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--band',type=int,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 g=gpu_guard();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.05)
 out=R/'runs'/('c116_threshold_'+str(a.band)+('_smoke' if a.smoke else ''));out.mkdir(exist_ok=False)
 data={};stats={}
 for part in ['fit','monitor']:
  t=load(part,'target');sel=np.flatnonzero(t[:,2+a.band]>0)
  if a.smoke:sel=sel[:65536]
  x=features(part,sel);y=t[sel,a.band]-load(part,'base')[sel,a.band]
  assert np.isfinite(x).all() and np.isfinite(y).all()
  weight=np.asarray(load(part,'weight')[sel],np.float32) if part=='fit' else None
  data[part]=Pool(x,y,weight=weight)
  stats[part]={'n':len(sel),'baseRMSE':float(np.sqrt(np.mean(y.astype('float64')**2)))}
 params=dict(iterations=20 if a.smoke else 3000,depth=10,learning_rate=.05,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=10,
  border_count=254,random_seed=116041,task_type='GPU',devices='0',gpu_ram_part=.70,thread_count=6,
  early_stopping_rounds=200,boosting_type='Plain',bootstrap_type='Bayesian',bagging_temperature=1.,boost_from_average=False,train_dir=str(out/'catboost'))
 atomic_json(out/'config.json',dict(GPU=g,params=params,stats=stats,inputs=169,official_val_used=False,
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 model=CatBoostRegressor(**params);start=time.monotonic()
 model.fit(data['fit'],eval_set=data['monitor'],use_best_model=True,verbose=100)
 model.save_model(str(out/'model.cbm'))
 delta=model.predict(data['monitor'],task_type='CPU',thread_count=6).astype('float32')
 err=delta-data['monitor'].get_label();rmse=float(np.sqrt(np.mean(err.astype('float64')**2)))
 np.save(out/'monitor_delta.npy',delta)
 atomic_json(out/'result.json',dict(status='complete',band=a.band,trees=model.tree_count_,best_iteration=model.get_best_iteration(),
  monitor_rmse=rmse,C82=stats['monitor']['baseRMSE'],gain=stats['monitor']['baseRMSE']-rmse,seconds=time.monotonic()-start,
  importance=model.get_feature_importance().tolist(),official_val_used=False,GPU=g,smoke=a.smoke))
 print('DONE',json.loads((out/'result.json').read_text()),flush=True)
if __name__=='__main__':main()
