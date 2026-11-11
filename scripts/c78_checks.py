"""Terrain-clustered uncertainty, independent CPU ray checks, and integration-resolution sensitivity."""
import csv,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c78_clear';sys.path.insert(0,str(R/'scripts'))
from c78_geometry_attribution import read_parts
from c78_counterfactual import make,MODELS

def interp(h,x,y,anti):
    xx=np.clip(x,0,255);yy=np.clip(y,0,255);i=np.minimum(xx.astype(int),254);j=np.minimum(yy.astype(int),254)
    u=xx-i;v=yy-j;a=h[j,i];b=h[j,i+1];c=h[j+1,i];d=h[j+1,i+1]
    if anti:return np.where(u+v<=1,a+(b-a)*u+(c-a)*v,d+(d-c)*(u-1)+(d-b)*(v-1))
    return np.where(v<=u,a+(b-a)*u+(d-b)*v,a+(d-c)*u+(c-a)*v)
def cpu_check(split,geo,panel):
    rr=list(csv.DictReader((R.parent/f'data/{split}_index.csv').open()));hh=np.load(R.parent/f'data/{split}_hm.npy',mmap_mode='r')
    rgen=np.random.default_rng(780977);ids=panel['ids'];pp=panel['pixels'];_,first=np.unique(panel['terrain'],return_index=True)
    selected=rgen.choice(first,64,replace=False);total=np.zeros(2,int);occluded=np.zeros(2,int);boundary_min=1e5
    lookup=[np.load(R/f'src/c70_centroid_lookup{a}.npz') for a in (0,1)]
    tris=[(x['ids'],x['weights']) for x in lookup]
    t=np.linspace(0,1,2049)[None,:]
    for j in selected:
        row=rr[ids[j]];h=hh[int(row['hm_row'])].astype('float64');tr,tc=int(row['tx_row']),int(row['tx_col'])
        sx=tc*255/256;sy=(tr+1)*255/256
        i=min(254,int(sx));l=min(254,int(sy));u=sx-i;v=sy-l;bi=(1-v)*((1-u)*h[l,i]+u*h[l,i+1])+v*((1-u)*h[l+1,i]+u*h[l+1,i+1])
        zchoices=np.floor([h[tr,tc],h[min(tr+1,255),tc],bi])+3
        for kind,mask in enumerate([geo[j,:,15]>.5,geo[j,:,:11].min(1)<.5]):
            sites=np.flatnonzero(mask)
            if not len(sites):continue
            px=np.repeat(pp[j,rgen.choice(sites,min(12,len(sites)),replace=False)],8)
            for anti,(idx,w) in enumerate(tris):
                v=rgen.random(len(px));part=(v[:,None]>np.cumsum(w[px],axis=1)).sum(1).clip(0,2)
                face=idx[px,part];cell=face%65025;second=face>=65025
                a=np.sqrt(rgen.random(len(px)));b=rgen.random(len(px));w0=1-a;w1=a*(1-b);w2=a*b
                if anti:fx=np.where(second,w0+w2,w1);fy=np.where(second,w0+w1,w2)
                else:fx=np.where(second,w1,w1+w2);fy=np.where(second,w1+w2,w2)
                xt=np.clip(cell%255+fx+rgen.uniform(-.5,.5,len(px)),0,255);yt=np.clip(cell//255+fy+rgen.uniform(-.5,.5,len(px)),0,255)
                zt=interp(h,xt,yt,anti)+rgen.uniform(.75,1.25,len(px))
                x0=np.clip(sx+rgen.uniform(-.5,.5,len(px)),0,255);y0=np.clip(sy+rgen.uniform(-.5,.5,len(px)),0,255);z0=rgen.uniform(min(zchoices)-.5,max(zchoices)+.5,len(px))
                x=x0[:,None]+(xt-x0)[:,None]*t;y=y0[:,None]+(yt-y0)[:,None]*t;ray=z0[:,None]+(zt-z0)[:,None]*t
                margin=(ray-interp(h,x,y,anti)).min(1);total[kind]+=len(px);occluded[kind]+=(margin<-.0002).sum()
                if kind==0:boundary_min=min(boundary_min,float(margin.min()))
    assert occluded[0]==0 and occluded[1]>0
    return dict(terrains=len(selected),independent_cpu_rays=total.tolist(),blocked_rays=occluded.tolist(),minimum_sampled_clearance_in_wide_certificate_m=boundary_min,samples_per_ray=2049,groups=['wide_certified','susceptible_positive_control'])
def run(split):
    panel=np.load(A/f'{split}_panel.npz');ref=np.load(A/f'{split}_reference.npy',mmap_mode='r');g,_=read_parts(split,'geom');p,_=read_parts(split,'power')
    o=np.empty((*ref.shape[:2],10),np.float32)
    for d in A.glob(f'{split}_sheet_*of*'):o[np.load(d/'positions.npy')]=np.load(d/'sheet.npy',mmap_mode='r')
    select=~panel['calibration'] if split=='train' else np.ones(len(ref),bool)
    unique,inv=np.unique(panel['terrain'][select],return_inverse=True)
    stats={}
    for b,f in enumerate([415e6,5.775e9]):
        clear=(ref[...,6]>.99)&(ref[...,4+b]>.5);err=ref[...,b]-ref[...,2+b];ss=err.astype('float64')**2
        deficit=ref[...,b]>20*np.log10(4*np.pi*f/299792458*g[...,17].clip(.001))+3
        groups=[clear,clear&(g[...,:11].min(2)<.9),clear&(g[...,:11].min(2)>=.99),clear&(g[...,15]>.5),clear&deficit,clear&(g[...,15]>.5)&deficit]
        matrix=np.stack([np.stack((q.sum(1),(q*ss).sum(1)),1) for q in groups],1)[select]
        summed=np.zeros((len(unique),len(groups),2));np.add.at(summed,inv,matrix);rng=np.random.default_rng(287011)
        boot=summed[rng.integers(len(summed),size=(2000,len(summed)))].sum(1);shares=boot/boot[:,:1]
        stats[str(b)]={name:dict(pixel_and_SSE_95=np.quantile(shares[:,j],[.025,.975],axis=0).T.tolist(),rmse95=np.quantile(np.sqrt(boot[:,j,1]/boot[:,j,0]),[.025,.975]).tolist()) for j,name in enumerate(['clear','switch90','stable99','wide','deficit3','wide_deficit3'])}
    ix=[];fine=[]
    for d in A.glob(f'{split}_power_*of*'):
        a=np.load(d/'sensitivity.npz');ix.extend(a['positions'].tolist());fine.extend(a['power'])
    ix=np.array(ix,dtype=int);fine=np.stack(fine);basep=p[ix];rr=ref[ix];gg=g[ix];oo=o[ix];res={}
    for b in (0,1):
        mask=(rr[...,6]>.99)&(rr[...,4+b]>.5)
        def db(v):return 10*np.log10(v.clip(1e-27))
        low=basep[...,2+b]+basep[...,6+b]+basep[...,14+b];high=fine[...,2+b]+fine[...,6+b]+fine[...,14+b]
        diff=(db(low)-db(high))[mask].astype('float64')
        coeff=np.array(json.load(open(A/'internal_coefficients.json'))[str(b)],np.float32)
        m=list(MODELS).index('geometry_power');xb,_,_,_=make(rr,gg,basep,oo,b,panel['ids'][ix],split);xf,_,_,_=make(rr,gg,fine,oo,b,panel['ids'][ix],split)
        e=(rr[...,b]-rr[...,2+b])[mask].astype('float64')
        cb=(xb@coeff[:,m]).clip(-20,20)[mask];cf=(xf@coeff[:,m]).clip(-20,20)[mask]
        rm=lambda v:float(np.sqrt(np.mean(v*v)))
        res[str(b)]=dict(clear_pixels=int(mask.sum()),physics_dB_quadrature_delta_RMS=rm(diff),physics_dB_delta_absolute_Q95=float(np.quantile(abs(diff),.95)),
                        learned_correction_delta_RMS=rm(cb-cf),C76_RMSE=rm(e),corrected_using_main_quadrature=rm(e-cb),corrected_using_finer_quadrature=rm(e-cf))
    result=dict(cluster_intervals=stats,numeric_sensitivity=dict(maps=len(ix),bands=res,comparison='spec 64 vs144 area points per face, diffuse 512+128 vs1536+384 with independent diffuse light sample seed'),geometry_check=cpu_check(split,g,panel))
    (A/f'{split}_uncertainty_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(split,result['geometry_check'],res,flush=True)
if __name__=='__main__':run(sys.argv[1])
