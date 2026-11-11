"""Positive control: does our finite-ray counter recover direct hits when the original solver's input IS known?"""
import csv,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c66_lattice import counts,Mesh
A=R/'audit/c79_receiver';C=R/'audit/c78_clear';D=R.parent/'data'
def main():
    mesh=Mesh(1,0);rows=list(csv.DictReader((D/'train_index.csv').open()))
    hs=np.load(D/'train_hm.npy',mmap_mode='r');pan=np.load(C/'train_panel.npz')
    ref=np.load(C/'train_reference.npy',mmap_mode='r')
    geom={}
    for shard in C.glob('train_geom_*of*'):
        arr=np.load(shard/'geometry.npy',mmap_mode='r')
        geom.update({int(i):(arr,j) for j,i in enumerate(np.load(shard/'positions.npy'))})
    sheet={};power={}
    for kind,which in [('sheet',sheet),('power',power)]:
        for sh in C.glob('train_'+kind+'_*of*'):
            z=np.load(sh/(kind+'.npy'),mmap_mode='r');which.update({int(i):(z,j) for j,i in enumerate(np.load(sh/'positions.npy'))})
    records=[];maps=[];t0=time.monotonic()
    for shard in sorted(A.glob('rt_*[0-9]')):
        for r in json.load(open(shard/'complete.json'))['records']:
            pos=r['pos'];index=r['index'];row=rows[index];h=hs[int(row['hm_row'])]
            tr,tc=int(row['tx_row']),int(row['tx_col']);arr,j=geom[pos]
            use=(arr[j,:,15]>.5)&(ref[pos,:,6]>.99)
            if not use.any():continue
            xp=pan['pixels'][pos][use]
            verts,inv,w=mesh.vertices(h,tr,tc,xp);zold=np.floor(h[tr,tc]);znew=np.floor(h[min(tr+1,255),tc])
            verts[...,2]+=float(zold-znew)
            # All compared arms have exactly the same scene, ray budget, released sampling locations and power scaling.
            sol=np.load(shard/(str(pos)+'.npz'))['features'][0,use]
            truth,coverage=sol[:,0],sol[:,1]
            yy=ref[pos,use,:2];valid=ref[pos,use,4:6]>.5
            e=(yy-ref[pos,use,2:4])
            dmax=arr[j,use,17];deficit=(yy-20*np.log10(4*np.pi*np.array([415e6,5.775e9])[None]/299792458*dmax[:,None]))>3
            record=[]
            for N,angle,dz in ((4_000_000,0,0.),(4_000_000,.013,0.),(4_000_000,0,.25),(4_000_000_000,0,0.),(4_000_000_000,.013,0.)):
                vv=verts.copy();vv[...,2]-=dz
                if angle:
                    xx=vv[...,0].copy();y=vv[...,1].copy()
                    vv[...,0]=xx*np.cos(angle)-y*np.sin(angle);vv[...,1]=xx*np.sin(angle)+y*np.cos(angle)
                pred=counts(vv,N,96)
                good=np.all((pred[:,2][inv]>=0)|(w<1e-6),axis=1)
                pow=(w*pred[:,0][inv]).sum(1);c=(w*(pred[:,2][inv]>0)).sum(1)
                lossdb=(-10*np.log10(pow.clip(1e-5))).clip(-35,50)
                record.append(np.stack([lossdb,c,good],1))
            pack=np.stack(record,1)
            sh,ksh=sheet[pos];pw,kpw=power[pos];ph=pw[kpw][use];extra=(ph[:,6:8]+ph[:,14:16])/ph[:,2:4].clip(1e-24)
            records.append((r,pack,np.stack([truth,coverage,valid[:,0],valid[:,1],deficit[:,0],deficit[:,1],e[:,0],e[:,1],sh[ksh,use,0],extra[:,0],extra[:,1],arr[j,use,18],arr[j,use,20]],1)))
            maps.append(r['terrain'])
            if len(records)%32==0:print('maps',len(records),'elapsed',round(time.monotonic()-t0,1),flush=True)
    out={'scope':'Official TRAIN only. Same registered mesh and Sionna 1.1.0 direct-only simulation. Evaluate certified interior unblocked points. Complete-case requires all weighted faces enumerated in every compared arm, per dataset and ray budget.',
        'synthetic_rays':4000000,'positions':'2048 preselected without replacement per map; only certified clear retained','groups':{}}
    for partition in ('select','confirm','all'):
        rec=[s for s in records if partition=='all' or s[0]['group']==partition]
        p=np.concatenate([s[1] for s in rec]);t=np.concatenate([s[2] for s in rec])
        exp={}
        # Synthetic target use same pixels across exact, phase-shifted and perturbed-height variants.
        keep=p[:,:3,2].min(1)>.5;obs=t[keep,0];cov=t[keep,1]
        evals={}
        for i,label in enumerate(('known_scene','wrong_launch_azimuth','wrong_TX_height')):
            z=p[keep,i,0];cv=p[keep,i,1]
            evals[label]=dict(direct_dB_RMSE=np.sqrt(np.mean((obs-z)**2)),direct_dB_correlation=np.corrcoef(obs,z)[0,1],
                              area_fraction_hit_MAE=np.mean(abs(cov-cv)),zero_raster_hit_agreement=np.mean((cv<1e-6)==(cov<1e-6)))
        exp['known_synthetic']=dict(n=len(obs),truth_zero_raster_fraction=np.mean(cov<1e-6),
            continuum_dB_RMSE=np.sqrt(np.mean(obs**2)),comparisons=evals)
        # Correlation with actual weak labels on the identical pixels. Rank amplitude and target indicator are different quantities.
        for N,ids in [(4000000,[0,1]),(4000000000,[3,4])]:
            keep=p[:,ids,2].min(1)>.5;rr={}
            for b in (0,1):
                valid=keep&(t[:,2+b]>.5);weak=t[valid,4+b].astype(bool);ll={}
                for j in ids:
                    risk=p[valid,j,0];zero=p[valid,j,1]<1e-6
                    if weak.sum() and (~weak).sum():
                        from scipy.stats import rankdata
                        auc=(rankdata(risk)[weak].sum()-weak.sum()*(weak.sum()+1)/2)/(weak.sum()*(~weak).sum())
                    else:auc=None
                    ll[str(j)]=dict(roc_auc=auc,weak_rate_given_zero=float(weak[zero].mean()) if zero.any() else None,zero_fraction=float(zero.mean()),weak_rate=float(weak.mean()))
                rr[str(b)]=dict(n=int(valid.sum()),scores=ll)
            exp['release_'+str(N)]=rr
        positive={}
        for name,sel in [('first',t[:,8]>.99),('previous',t[:,8]<.05)]:
            q=(p[:,:3,2].min(1)>.5)&sel
            gr={}
            for i in (0,1,2):
                zz=t[q,0];z=p[q,i,0];cov=t[q,1];cv=p[q,i,1]
                gr[str(i)]=dict(n=int(q.sum()),hit_fraction_MAE=float(np.mean(abs(cv-cov))),rms_direct_clipped=float(np.sqrt(np.mean((z-zz)**2))),
                                zero_hit_AGR=float(np.mean((cv<1e-6)==(cov<1e-6))))
            positive[name]=gr
        exp['first_versus_repeat_sheet_intersection']=positive
        from scipy.stats import rankdata
        def u(p,y):
            n=int(y.sum());m=len(y)-n
            if min(n,m)<2:return 0.,0
            return float(rankdata(p)[y].sum()-n*(n+1)/2),n*m
        st={}
        for task,scope,arm in [('synthetic',0,[0,1]),('415',0,[0,1]),('58',1,[0,1]),('415',0,[3,4]),('58',1,[3,4])]:
            for thin_background in ([False] if task=='synthetic' else [False,True]):
                vv=[]
                for row,pr,truth in rec:
                    ok=pr[:,arm,2].min(1)>.5
                    target=(truth[:,0]>3) if task=='synthetic' else truth[:,4+scope]>.5
                    if task!='synthetic':ok&=truth[:,2+scope]>.5
                    if thin_background:ok&=truth[:,9+scope]<.12
                    # Compare only receivers in the same terrain/TX, distance range and expected primary count.
                    bins=np.digitize(truth[:,11],[60,120,180])*4+np.digitize(truth[:,12],[20,100,1000])
                    sums=np.zeros(3)
                    for key in np.unique(bins[ok]):
                        q=(bins==key)&ok;a0,n=u(pr[q,arm[0],0],target[q]);a1,n1=u(pr[q,arm[1],0],target[q]);assert n==n1
                        sums+=[a0,a1,n]
                    vv.append(sums)
                vv=np.stack(vv);agg=vv.sum(0)
                if agg[2]==0:continue
                boots=vv[np.random.default_rng(852).integers(len(vv),size=(1500,len(vv)))].sum(1);den=boots[:,2].clip(1)
                dif=(boots[:,0]-boots[:,1])/den
                st[task+'_'+str(arm[0])+('_weak_extra' if thin_background else '_all')]=dict(within_stratum_auc_exact=float(agg[0]/agg[2]),
                    within_stratum_auc_wrong_phase=float(agg[1]/agg[2]),exact_minus_wrong_terrain95=np.quantile(dif,[.025,.975]).tolist(),positive_negative_pairs=int(agg[2]))
        exp['matched_phase']=st
        out['groups'][partition]=dict(maps=len(rec),**exp)
    (A/'known_scene_control.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
    c=out['groups']['confirm'];print('known scene',c['maps'],c['known_synthetic'],flush=True)
    print('actual',c['release_4000000000'],flush=True)
if __name__=='__main__':main()
