"""Tail-only conditional mean diagnostic. True-tail gating is explicitly oracle at scoring."""
import sys,json,time,argparse,hashlib,gc
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
import numpy as np,torch
from catboost import CatBoostRegressor,Pool
A=R/'audit/c125_tail';S=R/'audit/c101_residual';F=R/'audit/c120_sign'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['old','native'],required=True);a=ap.parse_args();g=gpu_guard();torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.03)
 out=R/f'runs/c125_tree_{a.mode}';out.mkdir(exist_ok=False)
 for b in range(2):
  pools={};meta={};labels={};weights={};indices={}
  for part in ['fit','monitor']:
   P=S/part;t=np.load(P/'target.npy',mmap_mode='r');base=np.load(P/'base.npy',mmap_mode='r');u=np.load(F/(part+'_rows.npy'));res=t[u,b]-base[u,b]
   keep=(t[u,b+2]>0)&(np.abs(res)>=5);ix=u[keep]
   x=np.c_[np.load(P/'common.npy',mmap_mode='r')[ix],np.load(P/'precise.npy',mmap_mode='r')[ix]]
   if a.mode=='native':x=np.c_[x,np.load(F/(part+'_native.npy'),mmap_mode='r')[keep]]
   x=np.asfortranarray(x,dtype='float32');y=res[keep].astype('float32');w=np.load(P/'weight.npy',mmap_mode='r')[ix] if part=='fit' else np.ones(len(ix),np.float32)
   pools[part]=Pool(x,y,weight=w if part=='fit' else None);labels[part]=y;weights[part]=w;indices[part]=ix
   allres=t[:,b].astype('float64')-base[:,b];valid=t[:,b+2]>0;tail=valid&(np.abs(allres)>=5)
   meta[part]=dict(n_tail=len(ix),valid=int(valid.sum()),C82_tail_RMSE=float(np.sqrt(np.mean(y.astype('float64')**2))),C82_all_SSE=float(np.sum(allres[valid]**2)),C82_nontail_SSE=float(np.sum(allres[valid&~tail]**2)))
  cfg=dict(iterations=3000,depth=10,learning_rate=.05,loss_function='RMSE',eval_metric='RMSE',l2_leaf_reg=10,border_count=254,random_seed=125041,task_type='GPU',devices='0',gpu_ram_part=.60,thread_count=6,early_stopping_rounds=200,boosting_type='Plain',bootstrap_type='Bayesian',bagging_temperature=1.,boost_from_average=False,train_dir=str(out/f'logs{b}'))
  atomic_json(out/f'config{b}.json',dict(params=cfg,mode=a.mode,GPU=g,inputs=169 if a.mode=='old' else 265,fit_and_score_scope='true-tail conditional diagnostic; not deployable',official_val_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data=meta))
  model=CatBoostRegressor(**cfg);t0=time.monotonic();model.fit(pools['fit'],eval_set=pools['monitor'],use_best_model=True,verbose=200);model.save_model(str(out/f'band{b}.cbm'))
  result=dict(mode=a.mode,band=b,trees=model.tree_count_,oracle_gate=True,official_val_used=False)
  for part in ['fit','monitor']:
   d=model.predict(pools[part],task_type='CPU',thread_count=6).astype('float32');y=labels[part].astype('float64');w=weights[part].astype('float64');correct=(d>=0)==(y>=0);pos=y>=0
   np.save(out/f'{part}_delta{b}.npy',d);np.save(out/f'{part}_rows{b}.npy',indices[part])
   tailSSE=float(np.sum((y-d)**2));oracleRMSE=float(np.sqrt((meta[part]['C82_nontail_SSE']+tailSSE)/meta[part]['valid']))
   result[part]=dict(tail_RMSE=float(np.sqrt(tailSSE/len(y))),weighted_tail_RMSE=float(np.sqrt(np.sum(w*(y-d)**2)/w.sum())),sign_accuracy=float(correct.mean()),balanced_sign_accuracy=float(.5*(correct[pos].mean()+correct[~pos].mean())),squared_residual_weighted_sign_accuracy=float(np.sum(correct*y*y)/np.sum(y*y)),truth_gated_all_RMSE=oracleRMSE,truth_gated_gain_vs_C82=float(np.sqrt(meta[part]['C82_all_SSE']/meta[part]['valid'])-oracleRMSE))
  result['seconds']=time.monotonic()-t0;atomic_json(out/f'result{b}.json',result);print('RESULT',result,flush=True)
  del pools,model;gc.collect()
 atomic_json(out/'complete.json',dict(status='complete',oracle_gate=True,official_val_used=False))
if __name__=='__main__':main()
