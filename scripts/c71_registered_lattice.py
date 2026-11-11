"""Test a geometry-sensitive direction-count hypothesis after discovering the adjacent-row TX height."""
import sys,os,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c66_lattice import Mesh,counts
from scipy.ndimage import uniform_filter
root=R/'audit/c66_transport';panel=json.load(open(root/'panels.json'))
old=set(json.load(open(root/'c68_grid_screen.json'))['indices'])|set(json.load(open(root/'c70_refine_grid_screen.json'))['indices'])
rng=np.random.default_rng(17385);ids=rng.choice(sorted(set(panel['fit_first'])-old),128,replace=False).tolist()
# Select input-defined line-of-sight sites on both panels to examine signed fine differences.
rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')));hm=np.load(R.parent/'data/train_hm.npy',mmap_mode='r')
labels=[np.load(R.parent/f'data/train_pl{b}.npy',mmap_mode='r') for b in ('415','58')]
mask=[np.load(R.parent/f'data/train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
ph=R/'features/c69_transport/part0of1'
ce=np.load(ph/'features.npy',mmap_mode='r');imap={i:j for j,i in enumerate(json.load(open(ph/'indices.json')))}
pr={}
for d in (R/'features/c66_frozen_maps').glob('fit*of2'):
    a=np.load(d/'prediction.npy',mmap_mode='r')
    for j,i in enumerate(json.load(open(d/'indices.json'))):pr[i]=(a,j)
ix=[];ys=[];ws=[];xs=[]
for i in ids:
    ft=ce[imap[i]].astype('float32');sel=np.flatnonzero((ft[10]>.998)&(uniform_filter(ft[10],3)>.98))
    if len(sel)<2048:sel=np.flatnonzero(ft[10]>.9)
    if not len(sel):sel=np.arange(65536)
    pix=rng.choice(sel,2048,replace=len(sel)<2048);ix.append(pix)
    p,k=pr[i];pred=p[k].reshape(2,-1)[:,pix].T
    ys.append(np.stack([l[i].ravel()[pix] for l in labels],1)-pred)
    ws.append(np.stack([np.unpackbits(m[i])[pix] for m in mask],1).astype(bool))
    xs.append(ft[[0,2,3,4,5,6,7,8,9,10]].reshape(10,-1)[:,pix].T)
ix=np.array(ix);xs=np.array(xs);ys=np.array(ys);ws=np.array(ws)
ns=[1000000,4000000,10000000,40000000,100000000,400000000,1000000000,4000000000]
phases=range(8);rec=[];start=time.monotonic()
for diag in (0,1):
    mesh=Mesh(1,diag)
    # Each scan uses 64 FIT terrains for coefficient fitting and 32 for configuration choice. The last 32 confirm only the selected configuration.
    for phase in phases:
        values=np.empty((len(ns),len(ids),2048,3),np.float32)
        for j,i in enumerate(ids):
            rr=rows[i];row,col=int(rr['tx_row']),int(rr['tx_col']);h=hm[int(rr['hm_row'])]
            vv,inv,w=mesh.vertices(h,row,col,ix[j]);vv[:,:,2]+=float(np.floor(h[row,col])-np.floor(h[min(row+1,255),col]))
            if phase&4:vv[:,:,[0,1]]=vv[:,:,[1,0]]
            if phase&1:vv[:,:,0]*=-1
            if phase&2:vv[:,:,1]*=-1
            for l,n in enumerate(ns):
                pred=counts(vv,n,96);a=pred[:,0][inv]
                values[l,j,:,0]=np.sum(w*np.log10(.02+a),1)-np.log10(1.02)
                values[l,j,:,1]=np.log10(.02+np.sum(w*a,1))-np.log10(1.02)
                values[l,j,:,2]=np.sum(w/(.02+pred[:,1][inv]),1).clip(0,30)
        for l,n in enumerate(ns):
            yc=values[l];vv=yc[:,:,:2].clip(-3,3);cnt=np.log1p(yc[:,:,2:]);weak=(np.maximum(xs[:,:,1:2],xs[:,:,3:4])<-.3)
            base=np.concatenate([np.ones((*cnt.shape[:2],1)),xs,cnt,cnt**2],axis=2)
            raw=np.concatenate([base,vv,vv*weak],axis=2)
            metrics=[]
            for b in [0,1]:
                rr=[]
                for feat in [base,raw]:
                    a=feat[:64][ws[:64,:,b]].astype('float64');t=ys[:64,:,b][ws[:64,:,b]]
                    beta=np.linalg.solve(a.T@a+np.eye(a.shape[1])*10,a.T@t)
                    err=(ys[:,:,b]-feat@beta)**2
                    rr.append([float(np.sqrt(err[sl][ws[sl,:,b]].mean())) for sl in [slice(0,64),slice(64,96),slice(96,None)]])
                metrics.append(rr)
            rec.append(dict(diag=diag,phase=phase,n=n,base_and_lattice=metrics))
        print(diag,phase,round(time.monotonic()-start,1),flush=True)
    (root/'c71_lattice_progress.json').write_text(json.dumps(dict(maps=ids,records=rec)))
best=max(rec,key=lambda r:sum(b[0][1]-b[1][1] for b in r['base_and_lattice']))
output=dict(scope='128 independent FIT terrains, 64 fit, 32 selection, 32 untouched FIT-side confirmation; stable LOS candidates',
    independent_maps=len(ids),selected=best,confirmation_gains=[b[0][2]-b[1][2] for b in best['base_and_lattice']],records=rec,elapsed=time.monotonic()-start)
(root/'c71_lattice_result.json').write_text(json.dumps(output,indent=2));print(json.dumps(output['selected']),output['confirmation_gains'])
