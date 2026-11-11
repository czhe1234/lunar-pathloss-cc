"""Independent point regression: geometry summaries plus surface material features.
CPU only; FIT labels are optimized, DEV only selects iteration. No C4 prediction input.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,json,time,hashlib,shutil
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[1];D=R/'features/c8_points';LABELS=R/'features/c6_points';p=argparse.ArgumentParser();p.add_argument('--variant',choices=['context'],required=True);p.add_argument('--threads',type=int,default=16);args=p.parse_args()
meta=json.loads((D/'metadata.json').read_text());n=len(meta['features'])
out=R/'runs'/f'c8_points_{args.variant}_s20261101';out.mkdir(exist_ok=True);assert not (out/'result.json').exists();(out/'source').mkdir(exist_ok=True)
for f in [Path(__file__),R/'scripts/c8_build_context_data.py',R/'src/c8_tx_context.py']:shutil.copy2(f,out/'source'/f.name)
params=dict(objective='regression',metric='rmse',learning_rate=.05,num_leaves=127,max_depth=12,min_data_in_leaf=100,max_bin=511,lambda_l2=1.,feature_fraction=1.,bagging_fraction=.85,bagging_freq=1,seed=20261101,num_threads=args.threads,device_type='cpu',deterministic=True,force_col_wise=True,verbosity=-1)
config={'variant':args.variant,'feature_count':n,'parameters':params,'max_trees':1200,'early_stopping_rounds':100,'point_data_metadata':meta,'lightgbm':lgb.__version__,'no_C4_or_C5_predictions_as_input':True,'official_validation_used':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
xx={s:np.load(D/f'{s}_x.npy',mmap_mode='r') for s in ['fit','dev']};yy={s:np.load(LABELS/f'{s}_y.npy',mmap_mode='r') for s in xx};fs={s:np.load(LABELS/f'{s}_fspl.npy',mmap_mode='r') for s in xx};mm={s:np.load(LABELS/f'{s}_mask.npy',mmap_mode='r') for s in xx};base=np.load(LABELS/'dev_baseline.npy',mmap_mode='r');results={};start=time.time()
for k,band in enumerate(['415','58']):
 keep={s:mm[s][:,k].astype(bool) for s in xx}
 train=lgb.Dataset(np.ascontiguousarray(xx['fit'][keep['fit'],:n]),label=(yy['fit'][:,k]-fs['fit'][:,k])[keep['fit']],feature_name=meta['features'][:n],free_raw_data=True)
 valid=lgb.Dataset(np.ascontiguousarray(xx['dev'][keep['dev'],:n]),label=(yy['dev'][:,k]-fs['dev'][:,k])[keep['dev']],reference=train,feature_name=meta['features'][:n],free_raw_data=True)
 history={};last=time.time()
 def progress(env):
  if env.iteration%100==0:
   rec={'variant':args.variant,'band':band,'iteration':env.iteration+1,'metrics':{f'{z[0]}_{z[1]}':z[2] for z in env.evaluation_result_list},'elapsed_seconds':time.time()-start}
   (out/'progress.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
 model=lgb.train(params,train,num_boost_round=1200,valid_sets=[train,valid],valid_names=['fit','dev'],callbacks=[lgb.early_stopping(100,verbose=False),lgb.record_evaluation(history),progress])
 model.save_model(str(out/f'band{band}.txt'),num_iteration=model.best_iteration)
 pred=model.predict(np.ascontiguousarray(xx['dev'][:,:n]),num_iteration=model.best_iteration,num_threads=args.threads).astype(np.float32)+fs['dev'][:,k]
 np.save(out/f'dev_pred{band}.npy',pred)
 mask=keep['dev'];err=pred[mask].astype(np.float64)-yy['dev'][mask,k].astype(np.float64);bc=base[mask,k].astype(np.float64)-yy['dev'][mask,k].astype(np.float64)
 gain=model.feature_importance(importance_type='gain');top=np.argsort(gain)[::-1][:20]
 result={'band':band,'best_iteration':int(model.best_iteration),'rmse_db':float(np.sqrt(np.mean(err**2))),'C4_rmse_on_same_sampled_pixels':float(np.sqrt(np.mean(bc**2))),'fit_valid_points':int(keep['fit'].sum()),'dev_valid_points':int(mask.sum()),'fit_rmse_at_selected_iteration':float(history['fit']['rmse'][model.best_iteration-1]),'seconds':time.time()-last,'top_features':[{'feature':meta['features'][j],'gain':float(gain[j])} for j in top]}
 results[band]=result;(out/f'band{band}_history.json').write_text(json.dumps(history)+'\n');(out/'partial_result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
 del model,train,valid
final={'state':'completed','variant':args.variant,'scope':'uniform sampled pixels from all FIT maps and the unchanged DEV maps; not full-map or official-validation score','bands':results,'seconds':time.time()-start,'no_GPU_used':True};(out/'result.json').write_text(json.dumps(final,indent=2)+'\n');(out/'progress.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
