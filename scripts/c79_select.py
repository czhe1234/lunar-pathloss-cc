"""Paired TRAIN-only screens for alternative receiver observations, followed by locked confirmation."""
import argparse, json, sys
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver'
STAGES=['all','clear','transition_shadow']
def read(p):return json.loads(p.read_text())
def load(stage):
    items=[];names=None
    for d in sorted(A.glob(stage+'[0-9]')):
        info=read(d/'complete.json');rows=info['records'];x=np.load(d/'x.npy',mmap_mode='r')
        if names is None:names=info['names']
        assert names==info['names'] and len(rows)==len(x)
        items.extend((row,x[k]) for k,row in enumerate(rows))
    items.sort(key=lambda rec:rec[0]['pos'])
    return [z[0] for z in items],np.stack([z[1] for z in items]),names

def common(ref):
    k=20*np.log10(5.775e9/415e6)
    d=(ref[...,3]-ref[...,2]-k).clip(-35,45)/12
    return np.stack([np.ones_like(d),d,np.maximum(d,0),np.maximum(d-.5,0),ref[...,6],1-ref[...,6],
                 (ref[...,2]-72)/20,(ref[...,3]-95)/20],-1)

def inputs(x,ref,b):
    a=x.astype('float32')
    phys=a[:,:,[b,2+b,4+b,6,7,8,9,10,11+b,13,14,15,16,17]]
    phys=phys/np.array([22,22,22,1,1,1,4,1,12,2,2,1,.2,1],np.float32)
    # Interactions allow the correction to differ in original-clear and shadow states.
    visible=ref[...,6:7]
    return np.concatenate([phys,phys*visible,phys*np.clip((ref[...,3:4]-ref[...,2:3]-23)/12,0,3)],2)

def fit_apply(xs,ref,b,fit,apply,beta=None):
    m=(ref[...,4+b]>.5);e=ref[...,b]-ref[...,2+b]
    if beta is None:
        zz=xs[fit][m[fit]].astype('float64');yy=e[fit][m[fit]].astype('float64')
        xt=zz.T@zz; lam=.004*np.maximum(np.diag(xt),len(zz)*.001);lam[0]=.01
        beta=np.linalg.solve(xt+np.diag(lam),zz.T@yy)
    off=(xs[apply]@beta.astype('float32')).clip(-24,24)
    return e[apply]-off,beta

def sse(err,ref,b):
    w=ref[...,4+b]>.5;cl=ref[...,6]>.99
    return np.stack([np.stack([np.sum(err.astype('float64')**2*q,axis=1),q.sum(1)],2-1) for q in (w,w&cl,w&~cl)],1)

def rms(z):
    return np.sqrt(z[...,0].sum(0)/z[...,1].sum(0))
def process(stage):
    rows,x,names=load(stage)
    ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r')[[r['pos'] for r in rows]]
    fit=np.array([r['group']=='fit' for r in rows]);apply=~fit;bas=common(ref);v0=names.index(['native','centroid'])
    if stage=='confirm':selected=read(A/'selected.json')
    mods={'C76':[],'shared':[],'native':[v0]}
    if stage=='screen':
        for j,v in enumerate(names):
            if j!=v0:mods['/'.join(v)]=[v0,j]
    else:
        for k,vs in selected['models'].items():mods[k]=[names.index(v) for v in vs]
    rec={};coefs={}
    def evaluate(label,jj):
        scores=[];cc=[]
        for b in (0,1):
            e=ref[...,b]-ref[...,2+b]
            if label=='C76':er=e[apply];beta=np.array([])
            else:
                layers=[bas]
                if jj:
                    xn=inputs(x[:,jj[0]],ref,b);layers.append(xn)
                    for j in jj[1:]:layers.append(inputs(x[:,j],ref,b)-xn)
                xx=np.concatenate(layers,2)
                trained=np.array(selected['coefs'][label][b]) if stage=='confirm' else None
                er,beta=fit_apply(xx,ref,b,fit,apply,trained)
            scores.append(sse(er,ref[apply],b));cc.append(beta.tolist())
        return np.stack(scores,1),cc
    for k,jj in mods.items():rec[k],coefs[k]=evaluate(k,jj)
    def vals(k):return rms(rec[k])
    if stage=='screen':
        from_each=[]
        for is_query in (True,False):
            keys=[k for k,v in mods.items() if len(v)==2 and (names[v[1]][0]=='native')==is_query]
            keys.sort(key=lambda k:float(vals(k)[:,0].mean()))
            from_each+=keys[:2]
        jj=[v0]+[mods[k][1] for k in from_each]
        name='combined';rec[name],coefs[name]=evaluate(name,jj);mods[name]=jj
        keepmodels={'shared':[],'native':[v0],name:jj,**{k:mods[k] for k in from_each}}
        out=dict(keep=[names[j] for j in dict.fromkeys(jj)],models={k:[names[j] for j in js] for k,js in keepmodels.items()},coefs={k:coefs[k] for k in keepmodels},selection_terrains=48,fit_terrains=80)
        (A/'selected.json').write_text(json.dumps(out)+'\n')
    base=vals('C76');native=vals('native');out={}
    for k,z in rec.items():
        out[k]=dict(rmse=vals(k).tolist(),gain=(base-vals(k)).tolist(),gain_vs_native=(native-vals(k)).tolist())
    (A/(stage+'_score.json')).write_text(json.dumps(dict(zones=STAGES,terrains=int(apply.sum()),results=out),indent=2)+'\n')
    np.savez_compressed(A/(stage+'_sse.npz'),names=list(rec),sse=np.stack(list(rec.values())),terrains=[r['terrain'] for r,b in zip(rows,apply) if b])
    print(stage,'base',base[:,:2].round(4).tolist(),'best',[(k,(base-vals(k))[:,0].round(5).tolist()) for k in sorted(rec,key=lambda k:vals(k)[:,0].mean())[:5]],flush=True)

if __name__=='__main__':process(sys.argv[1])
