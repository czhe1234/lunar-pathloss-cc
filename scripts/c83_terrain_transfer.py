"""Paired, held-out-transmitter test of fixed latent terrain error. This is a label-assisted diagnostic."""
from pathlib import Path
import sys,json,csv,time
import numpy as np
from scipy.ndimage import uniform_filter
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver'
def mm(p):return np.load(p,mmap_mode='r')
def main():
    rows=list(csv.DictReader((R.parent/'data/train_index.csv').open()))
    d=R/'runs/c54_combo_dev/bf16_compiled/all_shard0of1';indices=json.load(open(d/'indices.json'))['index_rows']
    base=[mm(d/f'pl{b}.npy') for b in ('415','58')];local=mm(R/'runs/c72_observed/full_internal/correction.npy')
    graph=mm(R/'runs/c73_wide_real/full_internal/prediction.npy')
    yall=[mm(R.parent/f'data/train_pl{b}.npy') for b in ('415','58')];wall=[mm(R.parent/f'data/train_mask{b}.npy') for b in ('415','58')]
    hmaps=mm(R.parent/'data/train_hm.npy')
    terrain=np.array([rows[i]['terrain_id'] for i in indices]);ty,tx=np.indices((256,256))
    total=[];keys=[];sample_counts=[];t0=time.monotonic()
    for name in sorted(set(terrain)):
        js=np.where(terrain==name)[0]
        if len(js)<10:continue
        unique={}
        for j in js:
            row=rows[indices[j]];unique.setdefault((int(row['tx_row']),int(row['tx_col'])),j)
        if len(unique)<10:continue
        js=np.array(list(unique.values()));rng=np.random.default_rng([8321,int(name)]);rng.shuffle(js)
        ids=np.array(indices)[js];coords=np.array([[int(rows[i][k]) for k in ('tx_row','tx_col')] for i in ids])
        h=hmaps[int(rows[ids[0]]['hm_row'])];sy,sx=np.gradient(h)
        dx=tx[None]-coords[:,1,None,None];dy=ty[None]-coords[:,0,None,None]
        rr=np.maximum(.25,np.sqrt(dx*dx+dy*dy));ux=(dx/rr).astype('float32');uy=(dy/rr).astype('float32')
        ztx=np.floor(h[(coords[:,0]+1).clip(0,255),coords[:,1]])+3
        dz=h[None]+1-ztx[:,None,None];dist=np.maximum(.25,np.sqrt(rr**2+dz**2))
        ca=(dz-sy[None]*dy-sx[None]*dx)/dist/np.sqrt(1+sx*sx+sy*sy)[None]
        grazing=np.exp(-abs(ca)/.025).astype('float32')
        fs=20*np.log10((4*np.pi/299792458.)*dist[:,None]*np.array([415e6,5.775e9])[None,:,None,None])
        p=.5*(np.stack([v[js] for v in base],1).astype('float32')+local[js].astype('float32')+graph[js].astype('float32'))
        yy=np.stack([v[ids] for v in yall],1).astype('float32')
        ww=np.stack([np.unpackbits(v[ids],axis=1).reshape(-1,256,256) for v in wall],1).astype('float32')
        e=yy-p;even=np.arange(len(js))%2==0;out=np.zeros((2,7,2))
        def filt(v):return uniform_filter(v,size=(3,3,1,1),mode='nearest')
        for hit in (even,~even):
            query=~hit
            for b in range(2):
                sf=w=ww[:,b]
                gradY,gradX=np.gradient(p[:,b],axis=(1,2))
                d=(p[:,b]-fs[:,b]).astype('float32')
                feat=np.stack([np.ones_like(ux),np.tanh(d/12),np.clip(gradX/4,-2,2),np.clip(gradY/4,-2,2),
                    grazing*ux,grazing*uy,np.sign(ca).astype('float32')*grazing],-1)
                x=feat[hit]*w[hit,:,:,None]**.5
                gram=np.einsum('txyi,txyj->xyij',x,x,optimize=True)
                ry=np.einsum('txyi,txy->xyi',x,e[hit,b]*w[hit],optimize=True)
                reg=np.eye(7)*max(3,len(x)*.17)
                co=np.linalg.solve(filt(gram)+reg,uniform_filter(ry,size=(3,3,1),mode='nearest')[...,None])[...,0]
                bb=ry[...,0]/(gram[:,:,0,0]+3)
                pred=(feat[query]*co[None]).sum(-1).clip(-12,12)
                # A mismatched-cell correction keeps each coefficient distribution but displaces its local registration.
                shco=np.roll(co,(17,31),axis=(0,1));wrong=(feat[query]*shco[None]).sum(-1).clip(-12,12)
                global_co=np.linalg.solve(gram.sum((0,1))+reg*65536,ry.sum((0,1)))
                global_pred=(feat[query]*global_co).sum(-1)
                # Match other transmitters in direction and predicted excess. No query label enters the estimate.
                angular=[]
                for q in np.flatnonzero(query):
                    align=(ux[hit]*ux[q]+uy[hit]*uy[q]).clip(-1,1)
                    weight=np.exp((align-1)*6-.5*np.abs(d[hit]-d[q])/5)*w[hit]
                    angular.append((weight*e[hit,b]).sum(0)/(weight.sum(0)+1.5))
                aq=np.stack(angular)
                variants=[0,bb[None],pred,wrong,global_pred,aq,.5*(pred+aq)]
                for i,z in enumerate(variants):
                    err=(e[query,b]-z).astype('float64');out[b,i]+=[np.sum(err**2*w[query]),w[query].sum()]
        total.append(out);keys.append(name);sample_counts.append(len(js))
        if len(keys)%10==0:print('terrains',len(keys),'seconds',round(time.monotonic()-t0,1),flush=True)
    results=np.stack(total);agg=results.sum(0);rm=np.sqrt(agg[...,0]/agg[...,1]);gain=rm[:,:1]-rm
    draw=np.random.default_rng(8362).integers(len(total),size=(2000,len(total)));s=results[draw].sum(1);br=np.sqrt(s[...,0]/s[...,1]);ci=np.quantile(br[:,:,:1]-br,[.025,.975],axis=0)
    labels=['C76','same_cell_bias','smoothed_terrain_geometry','displaced_cell_control','global_physics_control','match_direction_and_excess','geometry_plus_direction']
    out=dict(scope='Official TRAIN internal multi-TX terrains with >=10 distinct transmitters. Deduplicate H+TX, half the transmitters support and half query, swap halves. All variants with other TX labels are diagnostic-only.',terrains=len(keys),maps=sum(sample_counts),models=labels,rmse=rm.tolist(),gain=gain.tolist(),gain95=ci.tolist(),seconds=time.monotonic()-t0)
    (A/'terrain_cross_tx.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(A/'terrain_cross_tx.npz',sums=results,terrains=keys,counts=sample_counts)
    print(out['terrains'],out['maps'],np.round(gain,4).tolist(),flush=True)
if __name__=='__main__':main()
