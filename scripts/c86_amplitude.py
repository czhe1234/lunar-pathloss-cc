"""Separate azimuth-specific and phase-averaged direct power from what C76 already predicts."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver';C=R/'audit/c78_clear'
geo={};power={}
for k,lk in [('geom',geo),('power',power)]:
    for d in C.glob('train_'+k+'_*of*'):
        v=np.load(d/('geometry.npy' if k=='geom' else 'power.npy'),mmap_mode='r')
        lk.update({r:(v,j) for j,r in enumerate(np.load(d/'positions.npy'))})
ref=np.load(C/'train_reference.npy',mmap_mode='r');recs=[]
for p in sorted(A.glob('targeted*/*.npz')):
    t=np.load(p);i=int(t['pos']);loc=t['loc'];g,k=geo[i];g=g[k,loc];v,k=power[i];v=v[k,loc]
    raw=ref[i,loc];r=g[:,18].clip(.25)
    cov=t['values'];correct=cov[2,:,0]/r**2;shift=cov[3,:,0]/r**2
    freq=np.array([415e6,5.775e9]);constant=20*np.log10(4*np.pi*freq/299792458)
    allbands=[]
    for b in (0,1):
        w=raw[:,4+b]>.5;y=raw[:,b];base=raw[:,2+b];err=y-base
        extra=v[:,6+b]+v[:,14+b];continuum=v[:,2+b];avg=(correct+shift)/2
        pl=lambda v:constant[b]-10*np.log10(v.clip(1e-26))
        models=[base,pl(continuum+extra),pl(correct+extra),pl(shift+extra),pl(avg+extra),base+(pl(avg+extra)-pl(continuum+extra))]
        vals=np.array([np.sum((x[w]-y[w]).astype('float64')**2) for x in models])
        g0=pl(avg+extra)-pl(continuum+extra);g1=pl(correct+extra)-pl(shift+extra)
        # Bound the best one-coefficient effect on this panel, fitted on all of it only as a nondeployable diagnostic.
        def moment(v):return np.array([np.sum(v[w]*err[w],dtype='float64'),np.sum(v[w]**2,dtype='float64')])
        allbands.append(np.r_[w.sum(),vals,moment(g0),moment(g1)])
    recs.append(allbands)
s=np.array(recs);v=s.sum(0);r=np.sqrt(v[:,1:7]/v[:,:1])
def gain(rows):
    t=rows.sum(0);out=[]
    for b in (0,1):
        z=t[b];g=z[7]/max(z[8],1e-12)
        out.append(((z[1]/z[0])**.5-(max(0,z[1]-(z[7]*z[7])/max(z[8],1e-12))/z[0])**.5,g))
    return out
out=dict(scope='39 intentionally risk-stratified confirmation terrains, 1126 target pixels. Diagnostic only; these are not full-map scores. Constant gain shift relative to approximate first-order extra power.',
         variants=['C76','continuum+extra','correct-phase 4B+extra','shift-phase 4B+extra','mean over two phases+extra','C76 + full mean-phase power correction'],
         rmse=r.tolist(),most_favorable_same_panel_rms_gain_and_coefficient= gain(s),valid=v[:,0].astype(int).tolist())
(A/'targeted_amplitude.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
