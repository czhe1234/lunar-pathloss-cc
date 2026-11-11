"""Verify saved C43 predictions independently and compare matched terrain panels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json,csv
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c43_architecture_20261102'
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fit64','fit256'],required=True);p.add_argument('--fp32',action='store_true');a=p.parse_args()
proto=json.loads((A/'protocol.json').read_text())
rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
Y=[np.load(R.parent/'data'/f'train_pl{b}.npy',mmap_mode='r') for b in ['415','58']]
M=[np.load(R.parent/'data'/f'train_mask{b}.npy',mmap_mode='r') for b in ['415','58']]
models={};sources={}
for variant in ['cnn','grid','path']:
    run=R/'runs'/f'c43_{variant}_{a.stage}';res=json.loads((run/'result.json').read_text())
    predroot=run/'fp32' if a.fp32 else run
    meta=json.loads((predroot/'dev64_full.json').read_text());pr=np.load(predroot/'dev64_full.npy',mmap_mode='r')
    if a.fp32:
        res['DEV64_full']=meta
        res['FIT_panel_full']=json.loads((predroot/'fit_panel_full.json').read_text())
    assert meta['indices']==proto['dev64_ids'] and pr.shape==(len(meta['indices']),65536,2)
    ss=[];cc=[]
    for j,i in enumerate(meta['indices']):
        sums=[];counts=[]
        for b in range(2):
            keep=np.unpackbits(M[b][i],bitorder='big').reshape(256,256).astype(bool)
            pred=np.asarray(pr[j,:,b]).reshape(256,256)
            e=pred[keep].astype(np.float64)-Y[b][i][keep].astype(np.float64)
            sums.append(float(e@e));counts.append(int(keep.sum()))
        ss.append(sums);cc.append(counts)
    ss=np.array(ss);cc=np.array(cc);rm=np.sqrt(ss.sum(0)/cc.sum(0))
    assert np.max(np.abs(rm-np.array(res['DEV64_full']['RMSE'])))<1e-9
    assert cc.sum(0).tolist()==res['DEV64_full']['count']
    np.savez(predroot/'independent_dev64.npz',ids=meta['indices'],sse=ss,count=cc)
    by={}
    for j,i in enumerate(meta['indices']):
        t=rows[i]['terrain_id'];by.setdefault(t,[np.zeros(2),np.zeros(2)])
        by[t][0]+=ss[j];by[t][1]+=cc[j]
    tids=sorted(by)
    models[variant]={'RMSE':rm.tolist(),'combined':float(rm.mean()),
                     'FIT_panel_full':res['FIT_panel_full'],'best_step':res['best_step'],
                     'seconds':res['seconds'],'parameters':res['parameters'],'gpu_peak_GB':res['gpu_peak_GB'],
                     'mechanism':res.get('mechanism',{})}
    sources[variant]=(np.array([by[t][0] for t in tids]),np.array([by[t][1] for t in tids]))
configs=[json.loads((R/'runs'/f'c43_{v}_{a.stage}'/'config.json').read_text()) for v in ['cnn','grid','path']]
assert len({c['order_sha256'] for c in configs})==1
assert len({c['initial_state_sha256'] for c in configs})==1
rng=np.random.default_rng(20261102);ix=rng.integers(0,len(tids),(2000,len(tids)))
pairs={}
for new,ref in [('grid','cnn'),('path','cnn'),('path','grid')]:
    sn,cn=sources[new];sr,cr=sources[ref]
    delta=np.sqrt(sn[ix].sum(1)/cn[ix].sum(1))-np.sqrt(sr[ix].sum(1)/cr[ix].sum(1))
    observed=np.array(models[new]['RMSE'])-np.array(models[ref]['RMSE'])
    pairs[f'{new}_minus_{ref}']={'delta_RMSE_db':observed.tolist(),'CI95':np.quantile(delta,[.025,.975],axis=0).T.tolist(),
                                   'mean_delta':float(observed.mean()),'mean_CI95':np.quantile(delta.mean(1),[.025,.975]).tolist()}
winner=min(models,key=lambda k:models[k]['combined'])
scaling=None;promote=False
if a.stage=='fit256':
    previous=json.loads((A/('comparison_fit64_fp32.json' if a.fp32 else 'comparison_fit64.json')).read_text())['models']
    scaling={k:(np.array(previous[k]['RMSE'])-np.array(models[k]['RMSE'])).tolist() for k in models}
    for k in models:
        vs_c38=np.array(models[k]['RMSE'])<np.array(proto['C38_DEV64']['RMSE'])
        vs_cnn=np.array(models['cnn']['RMSE'])-np.array(models[k]['RMSE'])>=.1
        improved=np.array(scaling[k])>0
        if vs_c38.all() or (k!='cnn' and vs_cnn.all() and improved.all()):promote=True
out={'stage':a.stage,'precision':('FP32_no_TF32' if a.fp32 else 'original_BF16'),'scope':'frozen 64-terrain internal DEV,264 maps, all valid pixels; official TRAIN only',
     'models':models,'paired_terrain_bootstrap':pairs,'C38_same_DEV64':proto['C38_DEV64'],
     'winner_by_mean_band_RMSE':winner,'independent_score_verified':True,
     'same_map_order':True,'same_initial_weights':True,'scaling_improvement_db':scaling,'promote_full_FIT':promote,
     'limits':'One training seed, bootstrap conditions on these trained models; C38 used all FIT while these candidates used subsets; no official VAL scores'}
(A/f"comparison_{a.stage}{'_fp32' if a.fp32 else ''}.json").write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
