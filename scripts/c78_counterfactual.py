"""Frozen low-dimensional counterfactuals. Masks and true PL enter only fitting and evaluation."""
import json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear';sys.path.insert(0,str(R/'scripts'))
from c78_geometry_attribution import read_parts
REG=.002
MODELS={
 'C76':[], 'control':['C'], 'geometry':['C','G'], 'specular':['C','S0','S1'], 'diffuse':['C','L0','L1'],
 'power':['C','S0','S1','L0','L1'],'geom_spec':['C','G','S0','S1'],'geom_diff':['C','G','L0','L1'],
 'geometry_old_power':['C','G','S0','L0'],'geometry_power':['C','G','S0','S1','L0','L1'],
 'geometry_shuffled_power':['C','G','SP','LP'],'shuffled_geometry_power':['C','GP','S0','S1','L0','L1'],
 'geometry_sheet':['C','G','O'], 'geometry_sheet_power':['C','G','O','S0','S1','L0','L1'],
 'geometry_sheet_shuffled_power':['C','G','O','SP','LP']}
ZONES=['all','clear','switch','stable','altcert','widecert','deficit3','surplus3']
def lg(x):return 10*np.log10(np.maximum(x,1e-26))
def shuffled(key,seed):
    n,m=key.shape;out=np.empty((n,m),np.int32)
    for j in range(n):
        rng=np.random.default_rng(int(seed[j])+78119)
        rand=rng.random(m);order=np.lexsort((rand,key[j]));labels=key[j,order]
        cut=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,m]
        for a,b in zip(cut[:-1],cut[1:]):
            g=order[a:b];out[j,g]=np.roll(g,1)
    return out
def make(ref,g,p,o,band,ids,split):
    f=[415e6,5.775e9][band];K=20*np.log10(4*np.pi*f/299792458)
    clear=ref[...,6]>.99
    distance=g[...,18].clip(.25);fs=K+20*np.log10(distance);pred=ref[...,2+band];ex=(pred-fs).clip(-20,60)/5
    face=g[...,19].clip(0,1);nray=g[...,20].clip(0,1e8);lnd=np.log1p(distance)/5;lnc=np.log1p(nray)/7
    fine=g[...,:11].min(2);risk=1-fine
    # Strata are defined only by input or frozen prediction: same map, clear classification, distance, receiver angle,
    # geometry stability and coarse predicted excess. Random cyclic reassignment has no fixed point for non-singletons.
    key=((((clear.astype(np.int32)*4+np.digitize(distance,[60,120,180]))*3+np.digitize(face,[.015,.06]))*2+(fine<.90))*3+np.digitize(ex,[-.15,.6]))
    perm=shuffled(key,ids);reord=lambda v:np.take_along_axis(v,perm[...,None],axis=1)
    pack=lambda *a:np.stack(a,axis=-1).astype(np.float32)
    d0=p[...,band].clip(1e-21);direct=p[...,2+band].clip(1e-21);s0=p[...,4+band].clip(0);s1=p[...,6+band].clip(0);sf=p[...,8+band].clip(0)
    l0=p[...,10+band].clip(0);lr=p[...,12+band].clip(0);lm=p[...,14+band].clip(0);ll=p[...,16+band].clip(0)
    old=d0+s0+l0;ratio=lambda a,b:(lg(a)-lg(b)).clip(-40,40)/5
    c=pack(np.ones_like(ex),ex,np.maximum(ex,0),np.maximum(ex-1,0),lnd,lnd**2,lnc,np.maximum(.6-lnc,0),np.log1p(face*100)/4,
          ex*lnd,ex*(face<.025))
    loss=1-g[...,:13].clip(0,1)
    gd=ratio(old,d0*(1-.99*risk)+s0+l0)
    gg=pack(loss[...,1],loss[...,2],loss[...,3],loss[...,4],loss[...,5],loss[...,6],loss[...,7:11].max(2),loss[...,11],loss[...,12],
          risk,np.sqrt(risk.clip(0)),gd,gd*ex,1-g[...,13],1-g[...,14],1-g[...,15],np.log1p(g[...,17]-g[...,16]).clip(0,7)/3)
    se=ratio(direct+l0,direct+s0+l0);le=ratio(direct+s0,direct+s0+l0)
    lit=(1-risk);sr=ratio(old,direct+s1+l0);lf=ratio(old,direct+s0+lm);sfv=ratio(direct+s1+l0,direct+sf+l0)
    s0x=pack(se,ratio(s0+1e-5*direct,direct),se*ex,se*lnc)
    l0x=pack(le,ratio(l0+1e-5*direct,direct),le*ex,le*lnc)
    s1x=pack(sr,sfv,abs(sr),abs(sfv),sr*ex,sfv*ex,ratio(direct+l0,direct+s1+l0)*lit)
    l1x=pack(lf,ratio(direct+s0+lr,direct+s0+lm),ratio(direct+s0+ll,direct+s0+lm),abs(lf),lf*ex,ratio(direct+s1,direct+s1+lm)*lit,
             ((p[...,20+band]-(-lg(direct+s1+lm)))/5).clip(-8,8))
    obs1=ratio(old,d0*(.01+.99*o[...,0])+s0+l0);obs2=ratio(old,d0*(.01+.99*o[...,1])+s0+l0)
    ox=pack(1-o[...,0],1-o[...,1],np.log1p(o[...,2]),o[...,3],o[...,4]/7,o[...,5],o[...,6],o[...,7],o[...,8]/7,o[...,9],
            obs1,obs2,obs1*ex,obs2*ex,o[...,3]*ex,np.maximum(.6-o[...,8]/7,0)**2)
    blocks={'C':c,'G':gg,'S0':s0x,'S1':s1x,'L0':l0x,'L1':l1x,'O':ox,
            'SP':reord(np.concatenate((s0x,s1x),2)),'LP':reord(np.concatenate((l0x,l1x),2)),'GP':reord(gg)}
    slices={};start=0
    for k,v in blocks.items():slices[k]=np.arange(start,start+v.shape[-1]);start+=v.shape[-1]
    return np.concatenate(list(blocks.values()),axis=2),slices,clear,np.mean(perm!=np.arange(perm.shape[1])[None])
def target_bins(ref,g,b,clear,k):
    y=ref[...,b];valid=ref[...,4+b]>.5;v=g[...,:11].min(2);limit=k+20*np.log10(g[...,17].clip(.001));lower=k+20*np.log10(g[...,16].clip(.001))
    return np.stack([valid,valid&clear,valid&clear&(v<.90),valid&clear&(v>=.99),valid&clear&(g[...,14]>.5),valid&clear&(g[...,15]>.5),
                     valid&clear&(y>limit+3),valid&clear&(y<lower-3)],axis=2)
def run(split):
    panel=np.load(A/f'{split}_panel.npz');ref=np.load(A/f'{split}_reference.npy',mmap_mode='r');ids=panel['ids'];terrain=panel['terrain']
    g,_=read_parts(split,'geom');p,_=read_parts(split,'power')
    # Sheet files use their own name.
    parts=sorted(A.glob(f'{split}_sheet_*of*'));o=np.empty((*ref.shape[:2],10),np.float32)
    assert sum(len(np.load(d/'positions.npy')) for d in parts)==len(ids)
    for d in parts:
        assert (d/'complete.json').is_file();o[np.load(d/'positions.npy')]=np.load(d/'sheet.npy',mmap_mode='r')
    cal=panel['calibration'] if split=='train' else np.zeros(len(ids),bool)
    store=np.zeros((len(ids),2,len(ZONES),len(MODELS)+1));metric=np.zeros((len(ids),2,len(ZONES),10));start=time.monotonic();fixed={}
    for b in [0,1]:
        K=20*np.log10(4*np.pi*[415e6,5.775e9][b]/299792458)
        if split=='train':
            G=None
            ix=np.flatnonzero(cal)
            for at in range(0,len(ix),16):
                ii=ix[at:at+16];X,sl,c,_=make(ref[ii],g[ii],p[ii],o[ii],b,ids[ii],split)
                use=c&(ref[ii,:,4+b]>.5);m=X[use].astype(np.float64);y=(ref[ii,:,b]-ref[ii,:,2+b])[use].astype('float64')
                if G is None:G=np.zeros((m.shape[1],m.shape[1]));zy=np.zeros(m.shape[1]);ct=0
                G+=m.T@m;zy+=m.T@y;ct+=len(y)
            beta=np.zeros((len(zy),len(MODELS)))
            for j,(name,blocks) in enumerate(MODELS.items()):
                if not blocks:continue
                sel=np.concatenate([sl[k] for k in blocks]);gg=G[np.ix_(sel,sel)];reg=REG*np.maximum(np.diag(gg),ct*.0004);reg[0]=1e-8
                beta[sel,j]=np.linalg.solve(gg+np.diag(reg),zy[sel])
            fixed[str(b)]=beta.tolist()
        else:beta=np.array(json.load(open(A/'internal_coefficients.json'))[str(b)])
        beta=beta.astype('float32');perm_rates=[]
        for at in range(0,len(ids),16):
            s=slice(at,min(at+16,len(ids)));rr=ref[s];gg=g[s];pp=p[s];X,_,clear,rate=make(rr,gg,pp,o[s],b,ids[s],split)
            perm_rates.append(rate)
            e=rr[...,b]-rr[...,2+b];delta=(X@beta).clip(-20,20)*clear[:,:,None];sq=(e[:,:,None]-delta).astype('float64')**2
            masks=target_bins(rr,gg,b,clear,K).transpose(0,2,1).astype('float64')
            store[s,b,:,:-1]=masks@sq;store[s,b,:,-1]=masks.sum(2)
            d=pp[...,2+b];sp=pp[...,6+b];df=pp[...,14+b];y=rr[...,b]
            phys=lambda v:(K-lg(v)-y)**2
            dmax=K+20*np.log10(gg[...,17].clip(.001));noextra=np.maximum(y-dmax,0)
            # Cap by current loss to describe its overlap with a physical-model error lower bound.
            met=np.stack((phys(d),phys(d+sp),phys(d+df),phys(d+sp+df),phys(pp[...,b]+pp[...,4+b]+pp[...,10+b]),
              np.minimum(e*e,noextra**2),noextra**2,(y>dmax+3),((y>dmax+3)&(rr[...,2+b]>=dmax)),np.ones_like(y)),axis=2).astype('float64')
            metric[s,b]=masks@met
        print(split,'band',b,'matched-shuffle moved',np.mean(perm_rates),'seconds',round(time.monotonic()-start,1),flush=True)
    if split=='train':(A/'internal_coefficients.json').write_text(json.dumps(fixed,allow_nan=False)+'\n')
    np.savez_compressed(A/f'{split}_model_sse.npz',sse=store,physical=metric,models=list(MODELS),zones=ZONES,ids=ids,terrain=terrain)
    report={'split':split,'models':list(MODELS),'zones':ZONES,'ridge':REG}
    for name,select in ([('cal128',cal),('confirm484',~cal),('all612',np.ones(len(ids),bool))] if split=='train' else [('actual680',np.ones(len(ids),bool))]):
        v=store[select].sum(0);rms=np.sqrt(v[...,:-1]/v[...,-1:]);me=metric[select].sum(0);counts=me[...,-1]
        phy=np.sqrt(me[...,:5]/counts[:,:,None])
        terrains,inv=np.unique(terrain[select],return_inverse=True)
        # Two primary zones suffice for cluster confidence intervals for all nested comparisons.
        ss=np.zeros((len(terrains),2,2,len(MODELS)+1));np.add.at(ss,inv,store[select,:,:2])
        rng=np.random.default_rng(78237);boot=[]
        for _ in range(30):
            vv=ss[rng.integers(len(terrains),size=(100,len(terrains)))].sum(1);boot.append(np.sqrt(vv[...,:-1]/vv[...,-1:]))
        boot=np.concatenate(boot)
        detail={}
        for j,model in enumerate(MODELS):
            diff=boot[...,0]-boot[...,j]
            detail[model]=dict(rmse=rms[:,:,j].tolist(),gain=(rms[:,:,0]-rms[:,:,j]).tolist(),
                              full_clear_gain95=np.quantile(diff,[.025,.975],axis=0).transpose(1,2,0).tolist())
        contrasts={}
        names=list(MODELS)
        for left,right in [('geometry','control'),('power','control'),('geometry_power','geometry'),('geometry_power','power'),('geometry_power','geometry_shuffled_power'),('geometry_power','shuffled_geometry_power'),
                  ('geometry_power','geometry_old_power'),('geom_spec','geometry'),('geom_diff','geometry'),('geometry_power','geom_spec'),('geometry_power','geom_diff'),
                  ('geometry_sheet','geometry'),('geometry_sheet_power','geometry_sheet'),('geometry_sheet_power','geometry_power'),('geometry_sheet_power','geometry_sheet_shuffled_power')]:
            j,k=names.index(left),names.index(right);delta=boot[...,k]-boot[...,j]
            contrasts[left+'__vs__'+right]=dict(delta=(rms[:,:,k]-rms[:,:,j]).tolist(),full_clear95=np.quantile(delta,[.025,.975],axis=0).transpose(1,2,0).tolist())
        report[name]=dict(terrains=len(terrains),maps=int(select.sum()),counts=counts.astype(int).tolist(),results=detail,contrasts=contrasts,
           physics_RMSE_order=['direct','direct+spec64','direct+diffMIS','direct+both','old_quad_both'],raw_physics_RMSE=phy.tolist(),
           capped_direct_floor_overlap_share=(me[...,5]/v[:,:,0]).tolist(),
           positive_extra_only_class_floor_RMSE=np.sqrt(me[...,6]/counts).tolist(),
           data_weaker_than_direct_bound_over3=(me[...,7]/counts).tolist(),
           pred_already_weaker_among_deficit=(me[...,8]/np.maximum(me[...,7],1)).tolist())
    (A/f'{split}_counterfactual_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    z=report['confirm484' if split=='train' else 'actual680'];print('baseline, geometry, power, both, sheet',[(k,np.round(np.asarray(z['results'][k]['gain'])[:,:2],4).tolist()) for k in ['C76','control','geometry','power','geometry_power','geometry_sheet_power']],flush=True)
if __name__=='__main__':run(sys.argv[1])
