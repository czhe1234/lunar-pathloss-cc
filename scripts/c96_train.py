"""Matched CPU nonlinear readout: old features versus added dense lateral terrain strips."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
R=Path(__file__).resolve().parents[1];A=R/'audit/c96_information'
def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(part,extra):
    p=A/part;data={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ['x','y','mask','base','row_weight','mapid']}
    if extra:data['strip']=np.load(p/'strip.npy',mmap_mode='r')
    return data
def features(d,which,extra):
    x=np.array(d['x'][which],copy=True)
    if extra:x=np.concatenate([x,d['strip'][which]],1)
    return np.ascontiguousarray(x,dtype='float32')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['old','strip'],required=True);ap.add_argument('--band',type=int,choices=[0,1],required=True);ap.add_argument('--rounds',type=int,default=600);ap.add_argument('--threads',type=int,default=12);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    assert (A/'fit/done.json').exists() and (A/'dev/done.json').exists()
    extra=a.variant=='strip';out=A/f'{a.variant}_band{a.band}';out.mkdir(exist_ok=True);t=time.monotonic()
    fit=load('fit',extra);dev=load('dev',extra);dm=json.loads((A/'dev/done.json').read_text())
    group=np.repeat(dm['groups'],256);assert len(group)==len(dev['x'])
    train=np.flatnonzero(fit['mask'][:,a.band]>0);monitor=np.flatnonzero((dev['mask'][:,a.band]>0)&(group!='confirm'))
    if a.smoke:train=train[:4096];monitor=monitor[:4096]
    x=features(fit,train,extra);v=features(dev,monitor,extra)
    y=(fit['y'][train,a.band]-fit['base'][train,a.band]).astype(float);z=(dev['y'][monitor,a.band]-dev['base'][monitor,a.band]).astype(float)
    weight=np.array(fit['row_weight'][train]);weight/=weight.mean()
    init=float(np.sqrt(np.mean(z*z)))
    params=dict(objective='regression_l2',metric='rmse',learning_rate=.04,num_leaves=127,min_data_in_leaf=160,
                lambda_l2=20.,max_bin=255,feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,verbosity=-1,
                seed=960411,deterministic=True,force_col_wise=True,num_threads=a.threads,boost_from_average=False)
    cfg=dict(variant=a.variant,band=a.band,rows_fit=len(y),rows_monitor=len(z),features=x.shape[1],parameters=params,rounds=a.rounds,
             selection='min monitored maskedRMSE versus frozenC38 including zero correction; no confirmation labels used.',
             lightgbm=lgb.__version__,initial_monitor_rmse=init,CPU=True)
    if not a.smoke:write(out/'config.json',cfg)
    tr=lgb.Dataset(x,label=y,weight=weight,params=params,free_raw_data=True)
    va=lgb.Dataset(v,label=z,reference=tr,free_raw_data=True)
    history={}
    def progress(env):
        iteration=env.iteration+1
        if iteration%25==0 or iteration==1:
            val=float(env.evaluation_result_list[0][2]);rec=dict(iteration=iteration,rounds=a.rounds,monitor_RMSE=val,seconds=time.monotonic()-t)
            if not a.smoke:write(out/'progress.json',rec)
            print(a.variant,a.band,rec,flush=True)
    net=lgb.train(params,tr,num_boost_round=4 if a.smoke else a.rounds,valid_sets=[va],valid_names=['monitor'],
       callbacks=[lgb.record_evaluation(history),lgb.early_stopping(80,first_metric_only=True,verbose=False),progress])
    if a.smoke:
        pred=net.predict(v,num_threads=a.threads);assert np.isfinite(pred).all()
        print('smoke passed',cfg['features'],flush=True);return
    best=int(net.best_iteration);score=float(net.best_score['monitor']['rmse']);selected=best if score<init else 0
    net.save_model(str(out/'model.txt'),num_iteration=best)
    write(out/'history.json',history);write(out/'result.json',dict(status='complete',selected_iteration=selected,best_boost_round=best,
           initial_monitor_rmse=init,selected_monitor_rmse=min(init,score),gain_vs_C38=init-min(init,score),seconds=time.monotonic()-t))
    print('DONE',a.variant,a.band,min(init,score),'selected',selected,'seconds',round(time.monotonic()-t,1),flush=True)
if __name__=='__main__':main()
