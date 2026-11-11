"""Evaluate only previously frozen native-cell dB-domain diagnostic predictions."""
import json
from pathlib import Path
import numpy as np
import sys
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss';D=R.parent/'data'
sys.path.insert(0,str(R/'scripts'));from c95_score import stat,average_precision_score
p=json.loads((A/'plan.json').read_text());freeze=json.loads((A/'freeze.json').read_text());cfg=freeze['configs']
assert 'eta_frozen_unix' in freeze
for e in p['confirm']:
    assert all((A/'confirm'/f"{e['index']}_{c}.npz").exists() for c in cfg)
f=np.load(P/'patch_features.npz');b=np.load(P/'patch_target.npz');names=f['names'].tolist()
yy=[np.load(D/f'train_pl{k}.npy',mmap_mode='r') for k in ('415','58')];ww=[np.load(D/f'train_mask{k}.npy',mmap_mode='r') for k in ('415','58')]
collect=[];native={c:[] for c in cfg};weights={c:[] for c in cfg};availability=[]
const=20*np.log10(4*np.pi*np.array([415e6,5.775e9])/299792458)
for e in p['confirm']:
    j=e['patch_map'];i=e['index'];ix=np.concatenate([np.arange(z*49,(z+1)*49) for z in e['windows']]);px=f['pixels'][j,ix]
    y=np.stack([z[i].ravel()[px] for z in yy],1).astype(float);v=np.stack([np.unpackbits(z[i])[px] for z in ww],1).astype(bool)
    weak=(y-np.maximum(b['c82'][j,ix],b['c92'][j,ix])>=5)&v;clear=f['x'][j,ix,names.index('state')]==0
    collect.append(dict(y=y,v=v,weak=weak,fs=f['fs'][j,ix],clear=clear,base=b['c82'][j,ix],old=b['c92'][j,ix],terrain=np.repeat(i,len(px)),pixel=px))
    good=[]
    for c in cfg:
        z=np.load(A/'confirm'/f'{i}_{c}.npz');assert np.array_equal(z['pixel'],px);native[c].append(z['native']);weights[c].append(z['weights']);good.append(z['valid'])
    availability.append(np.logical_and.reduce(good))
data={k:np.concatenate([d[k] for d in collect]) for k in collect[0]};support=np.concatenate(availability)
fs=data['fs'];r2=10**((fs[:,0]-const[0])/10);pred={};scores={}
for c in cfg:
    n=np.concatenate(native[c])*r2[:,None];w=np.concatenate(weights[c]);eta=freeze['eta'][c];pp=[]
    for band in (0,1):
        pp.append(fs[:,band]+(-10*np.log10(n+eta[band])*w).sum(1))
    pred[c]=np.stack(pp,1);bs=[]
    for band in (0,1):
        m=data['v'][:,band]&data['clear']&support;weak=data['weak'][:,band]&m;normal=m&~weak
        residual=pred[c][:,band]-data['y'][:,band];excess=pred[c][:,band]-fs[:,band]
        alarm=m&(excess>3);tp=alarm&data['weak'][:,band]
        bs.append(dict(eta=eta[band],clear=stat(data['y'][:,band],pred[c][:,band],m),
             weak=stat(data['y'][:,band],pred[c][:,band],weak),normal=stat(data['y'][:,band],pred[c][:,band],normal),
             C82_clear=stat(data['y'][:,band],data['base'][:,band],m),C82_weak=stat(data['y'][:,band],data['base'][:,band],weak),
             C92_clear=stat(data['y'][:,band],data['old'][:,band],m),
             AP=float(average_precision_score(data['weak'][m,band],excess[m])),
             alarms=int(alarm.sum()),true_positive=int(tp.sum()),precision=float(tp.sum()/max(1,alarm.sum())),recall=float(tp.sum()/max(1,weak.sum()))))
    scores[c]=bs
np.savez_compressed(A/'confirm_native_predictions.npz',**data,**{'p_'+c:pp for c,pp in pred.items()},support=support)
out=dict(maps=len(p['confirm']),points=len(data['y']),scores=scores,source='Frozen H/TX-only original direct ray replay plus diagnostic scalar floor, native-cell dB interpolation. Not official score or solver-verified total field; no confirmation selection.')
(A/'confirm_native_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print('maps',out['maps'],'points',out['points'])
for c,bs in scores.items():print(c,[[round(z['AP'],4),round(z['clear']['RMSE'],3),round(z['weak']['MAE'],3),round(z['precision'],3),round(z['recall'],3)] for z in bs])
