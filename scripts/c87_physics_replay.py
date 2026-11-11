"""Apply previously frozen TRAIN-only diagnostic coefficients to C82 without refitting on validation."""
import json,hashlib
import numpy as np
from c87_read import ROOT,OUT,maps,write
from c87_sample import parts
from c78_counterfactual import make,MODELS
C=ROOT/'audit/c78_clear'
def main():
    pan=np.load(C/'val_panel.npz');ref=np.load(C/'val_reference.npy',mmap_mode='r')
    rec=np.array(ref,copy=True)
    for i,row,y,p,w in maps():
        k=pan['pixels'][i];rec[i,:,2:4]=p.reshape(2,-1)[:,k].T
    geo,pwr,sht=[parts(k) for k in ('geom','power','sheet')]
    def take(d,ids):return np.stack([d[int(i)][0][d[int(i)][1]] for i in ids])
    path=C/'internal_coefficients.json';ha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.load(open(C/'internal_decision.json'))['coefficients_sha256']==ha
    bet=json.load(open(path));names=['C76','geometry','geom_spec','geom_diff','geometry_power','geometry_shuffled_power']
    ix=[list(MODELS).index(n) for n in names];scores=np.zeros((len(rec),2,2,len(names)+1))
    for start in range(0,len(ref),16):
        ids=np.arange(start,min(len(ref),start+16));rr=rec[ids];g=take(geo,ids);p=take(pwr,ids);s=take(sht,ids)
        for b in (0,1):
            x,_,cl,_=make(rr,g,p,s,b,ids,'val');off=(x@np.array(bet[str(b)],np.float32)[:,ix]).clip(-20,20)*cl[:,:,None]
            assert np.max(abs(off[:,:,0]))==0
            errs=((rr[...,b]-rr[...,2+b])[:,:,None]-off).astype('float64')**2
            valid=rr[...,4+b]>.5;mask=np.stack((valid,valid&cl),1).astype('float64')
            scores[ids,b,:,:-1]=mask@errs;scores[ids,b,:,-1]=mask.sum(2)
    v=scores.sum(0);r=np.sqrt(v[...,:-1]/v[...,-1:]);out={}
    for j,n in enumerate(names):
        out['C82' if j==0 else n]=dict(rmse=r[...,j].tolist(),gain_vs_C82=(r[...,0]-r[...,j]).tolist())
    ut,inv=np.unique(pan['terrain'],return_inverse=True);by=np.zeros((len(ut),*scores.shape[1:]));np.add.at(by,inv,scores)
    boot=by[np.random.default_rng(8733).integers(len(ut),size=(1500,len(ut)))].sum(1);rr=np.sqrt(boot[...,:-1]/boot[...,-1:])
    j=names.index('geometry_power');m=names.index('geometry');delta=rr[...,m]-rr[...,j]
    write('physics.json',dict(scope='Frozen C78 coefficients fitted on TRAIN C76 research counterpart applied unchanged to C82 as a diagnostic on the fixed official validation panel. These corrections are not a C82-trained model or an official full score. Reflectance tests are first-order only.',coefficient_sha=ha,zones=['all','original_clear'],results=out,
        additional_gain_both_powers_vs_geometry=(r[...,m]-r[...,j]).tolist(),additional_gain95=np.quantile(delta,[.025,.975],axis=0).tolist(),
        real_vs_shuffled_power_gain=(r[...,names.index('geometry_shuffled_power')]-r[...,j]).tolist()))
    print('physics increment',r[...,m]-r[...,j])
if __name__=='__main__':main()
