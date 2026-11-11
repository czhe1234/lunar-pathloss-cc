"""Paired terrain bootstrap for frozen diagnostic models. Never select on confirmation."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';z=np.load(A/'confirm_native_predictions.npz')
groups,inv=np.unique(z['terrain'],return_inverse=True);rng=np.random.default_rng(950561);B=500
counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=B);out={}
for c in ['origin_xy128','origin_z128','mesh_delaunay','pixel_x_-0.5']:
    bands=[]
    for b in (0,1):
        m=z['v'][:,b]&z['clear']&z['support'];grp=inv[m];y=z['y'][m,b];weak=z['weak'][m,b];fs=z['fs'][m,b]
        old=z['p_base'][m,b];new=z['p_'+c][m,b]
        def apset(p):
            score=p-fs;ix=np.argsort(-score,kind='stable');ends=np.r_[np.flatnonzero(score[ix][1:]!=score[ix][:-1]),len(ix)-1]
            return ix,ends
        s0,s1=apset(old),apset(new)
        def ap(w,pair):
            ix,end=pair;tp=np.cumsum(w[ix]*weak[ix])[end];allw=np.cumsum(w[ix])[end];precision=np.divide(tp,allw,out=np.zeros_like(tp),where=allw>0)
            return float(np.sum(np.diff(np.r_[0,tp])*precision)/max(1,tp[-1]))
        vals=[]
        for ct in np.r_[np.ones((1,len(groups))),counts]:
            w=ct[grp];n=w.sum()
            gain=np.sqrt(np.sum(w*(old-y)**2)/n)-np.sqrt(np.sum(w*(new-y)**2)/n)
            vals.append([gain,ap(w,s1)-ap(w,s0)])
        a=np.asarray(vals);bands.append(dict(RMSE_gain=float(a[0,0]),RMSE_gain_CI95=np.quantile(a[1:,0],[.025,.975]).tolist(),
          AP_gain=float(a[0,1]),AP_gain_CI95=np.quantile(a[1:,1],[.025,.975]).tolist()))
    out[c]=bands
(A/'confirm_bootstrap.json').write_text(json.dumps(dict(terrains=len(groups),replicates=B,seed=950561,models=out),indent=2)+'\n');print(json.dumps(out))
