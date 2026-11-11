"""Controlled residual fit: original C10 inputs vs added receiver neighborhood geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np,lightgbm as lgb,json,time,shutil,hashlib
R=Path(__file__).resolve().parents[1];D=R/'features/c6_points';C=R/'features/c10_points';N=R/'features/c14_receiver_points';O=R/'runs/c14_receiver_residual_probe_s20261101';O.mkdir(exist_ok=False);(O/'source').mkdir()
for p in [Path(__file__),R/'scripts/c14_build_data.py',R/'src/c14_receiver_geometry.py',R/'audit/c14_protocol.json']:shutil.copy2(p,O/'source'/p.name)
xx={p:np.load(C/f'{p}_x.npy',mmap_mode='r') for p in ['fit','dev']};nn={p:np.load(N/f'{p}_x.npy',mmap_mode='r') for p in xx};yy={p:np.load(D/f'{p}_y.npy',mmap_mode='r') for p in xx};mask={p:np.load(D/f'{p}_mask.npy',mmap_mode='r').astype(bool) for p in xx}
base={'fit':np.load(R/'audit/c13_rx_geometry_probe/fit_c10_prediction.npy',mmap_mode='r'),'dev':np.column_stack([np.load(R/f'runs/c10_points_phase_geometry_s20261101/dev_pred{b}.npy') for b in ['415','58']])}
names=json.loads((C/'metadata.json').read_text())['features'];newnames=json.loads((N/'metadata.json').read_text())['features']
params=dict(objective='regression',metric='rmse',learning_rate=.05,num_leaves=63,max_depth=8,min_data_in_leaf=100,max_bin=511,lambda_l2=1.,feature_fraction=1.,bagging_fraction=.85,bagging_freq=1,seed=20261101,num_threads=16,device_type='cpu',deterministic=True,force_col_wise=True,verbosity=-1)
(O/'config.json').write_text(json.dumps({'parameters':params,'max_trees':400,'patience':60,'FIT_residuals_from_frozen_in_sample_C10':True,'same_DEV_points_as_C10':True,'official_VAL_used':False},indent=2)+'\n')
results={};start=time.time()
for variant in ['same_inputs','RX_observation']:
 vx={p:np.ascontiguousarray(xx[p]) if variant=='same_inputs' else np.concatenate((xx[p],nn[p]),1) for p in xx}
 fn=names if variant=='same_inputs' else names+newnames;results[variant]={};vdir=O/variant;vdir.mkdir()
 for k,b in enumerate(['415','58']):
  residual={p:(yy[p][:,k].astype(np.float32)-base[p][:,k]) for p in xx}
  train=lgb.Dataset(vx['fit'][mask['fit'][:,k]],label=residual['fit'][mask['fit'][:,k]],feature_name=fn,free_raw_data=True)
  valid=lgb.Dataset(vx['dev'][mask['dev'][:,k]],label=residual['dev'][mask['dev'][:,k]],reference=train,feature_name=fn,free_raw_data=True)
  hist={}
  def progress(env):
   if env.iteration%100==0:
    v={'variant':variant,'band':b,'iteration':env.iteration+1,'metrics':{z[0]:z[2] for z in env.evaluation_result_list},'seconds':time.time()-start};(O/'progress.json').write_text(json.dumps(v)+'\n');print(json.dumps(v),flush=True)
  model=lgb.train(params,train,num_boost_round=400,valid_sets=[train,valid],valid_names=['fit','dev'],callbacks=[lgb.early_stopping(60,verbose=False),lgb.record_evaluation(hist),progress])
  model.save_model(str(vdir/f'band{b}.txt'),num_iteration=model.best_iteration)
  pr=model.predict(vx['dev'],num_threads=16,num_iteration=model.best_iteration).astype(np.float32)+base['dev'][:,k];np.save(vdir/f'dev_pred{b}.npy',pr)
  m=mask['dev'][:,k];err=pr[m].astype(float)-yy['dev'][m,k].astype(float);before=base['dev'][m,k].astype(float)-yy['dev'][m,k].astype(float);rmse=float(np.sqrt(np.mean(err**2)));old=float(np.sqrt(np.mean(before**2)))
  importance=model.feature_importance(importance_type='gain');top=np.argsort(importance)[::-1][:15]
  results[variant][b]={'rmse_db':rmse,'C10_rmse_db':old,'relative_RMSE_reduction_percent':100*(1-rmse/old),'best_iteration':model.best_iteration,'valid_pixels':int(m.sum()),'top_features':[{'name':fn[j],'gain':float(importance[j])} for j in top]}
  (vdir/f'band{b}_history.json').write_text(json.dumps(hist)+'\n');(O/'partial_result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({variant:{b:results[variant][b]}}),flush=True)
  del model,train,valid
 del vx
promote=all(results['RX_observation'][b]['relative_RMSE_reduction_percent']>=5 and results['RX_observation'][b]['relative_RMSE_reduction_percent']-results['same_inputs'][b]['relative_RMSE_reduction_percent']>=1 for b in ['415','58'])
out={'state':'completed','scope':'fixed sampled internal DEV only; FIT residual calibrators; not official/full-map score','results':results,'promote_receiver_branch':promote,'seconds':time.time()-start,'GPU_used':False};(O/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
