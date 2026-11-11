"""Label-assisted, two-fold spatial controls. No validation correction is used in deployment."""
import sys,time
import numpy as np
from c87_read import OUT,maps,write
SHIFTS=[(0,0)]+[(dy,dx) for dy in range(-2,3) for dx in range(-2,3) if dx or dy]
SIZES=(256,16,8)
def blocks(a):
    return a.reshape(*a.shape[:-2],32,8,32,8).sum(axis=(-3,-1),dtype=np.float64)
def merge(a,size):
    f=size//8;s=32//f
    return a.reshape(*a.shape[:-2],s,f,s,f).sum(axis=(-3,-1))
def gather(a,ix):
    return np.take_along_axis(a,ix[None],axis=0)[0]
def evaluate(sum_e,sum_s,masks,size):
    # Tensors have shape [shift,fold,band,tile-row,tile-col].
    er=merge(sum_e,size);ss=merge(sum_s,size);n=merge(masks,size);out=np.zeros((2,5));moved=np.zeros((2,2))
    for k in (0,1):
        l=1-k;fit_n=n[k];test_n=n[l];ok=fit_n>=4
        di=np.where(ok,np.argmin(ss[:,k],axis=0),0)
        fit_offset=er[:,k]/np.maximum(fit_n,1)
        fit_offset*=ok
        dj=np.where(ok,np.argmin(ss[:,k]-fit_offset*er[:,k],axis=0),0)
        base=ss[0,l]
        vals=[base,gather(ss[:,l],di),base-2*fit_offset[0]*er[0,l]+fit_offset[0]**2*test_n,
              gather(ss[:,l]-2*fit_offset*er[:,l]+fit_offset**2*test_n,dj),test_n]
        out+=np.stack([v.sum((1,2)) for v in vals],1)
        moved[:,0]+=(test_n*(di!=0)).sum((1,2));moved[:,1]+=test_n.sum((1,2))
    return out,moved
def run(shard,parts):
    zmap=np.load(OUT/'visibility_class.npy',mmap_mode='r')
    Y,X=np.indices((256,256));inside=(X>=2)&(X<254)&(Y>=2)&(Y<254)
    parity=(X+Y)%2;results=[];moves=[];oracle=[];ids=[];t0=time.monotonic()
    for i,row,y,p,w in maps(lambda i:shard*2330//parts<=i<(shard+1)*2330//parts):
        w=w&inside[None];fold=np.stack([w&(parity==v) for v in (0,1)])
        ms=blocks(fold)
        er=np.empty((25,2,2,32,32));ss=np.empty_like(er)
        near_err=None;base_err=None
        for j,(dy,dx) in enumerate(SHIFTS):
            shifted=np.roll(p,shift=(-dy,-dx),axis=(1,2)) if j else p
            err=(shifted.astype('float64')-y)
            ew=fold*err[None];er[j]=blocks(ew);ss[j]=blocks(ew*ew)
            if j==0:near_err=abs(err);base_err=near_err.copy()
            else:np.minimum(near_err,abs(err),out=near_err)
        zv=zmap[i];oc=np.zeros((2,4,3,4))
        for b in (0,1):
            for zone,mask in enumerate([w[b],w[b]&(zv==0),w[b]&(zv==1),w[b]&(zv==2)]):
                for k,threshold in enumerate((0,5,10)):
                    q=mask&(base_err[b]>=threshold);ox=near_err[b][q]
                    oc[b,zone,k]=[q.sum(),(base_err[b][q]**2).sum(),(ox**2).sum(),np.count_nonzero(ox<=1)]
        per=[];mv=[]
        for sz in SIZES:
            stats,sw=evaluate(er,ss,ms,sz);per.append(stats);mv.append(sw)
        ids.append(i);oracle.append(oc);results.append(per);moves.append(mv)
        if len(ids)%200==0:print(shard,len(ids),round(time.monotonic()-t0,1),flush=True)
    np.savez(OUT/f'shift_part{shard}.npz',indices=ids,stats=results,moved=moves,oracle=oracle)
def finish(parts):
    ids=[];stats=[];moves=[];oracle=[]
    for k in range(parts):
        f=np.load(OUT/f'shift_part{k}.npz');ids.extend(f['indices']);stats.extend(f['stats']);moves.extend(f['moved']);oracle.extend(f['oracle'])
    assert ids==list(range(2330));a=np.stack(stats);u=a.sum(0)
    assert np.all(u[...,4]==u[0,:,4][None,:]);assert np.all(u[...,0]==u[0,:,0][None,:]) or np.allclose(u[...,0],u[0,:,0][None,:],rtol=1e-10)
    total=np.stack(oracle).sum(0);np.testing.assert_allclose(total[:,0,0,1],u[0,:,0],rtol=1e-9)
    r=np.sqrt(u[...,:4]/u[...,4:]);mv=np.stack(moves).sum(0)
    terrain=np.load(OUT/'full_by_map.npz')['terrain'];tn,inv=np.unique(terrain,return_inverse=True)
    by=np.zeros((len(tn),*a.shape[1:]));np.add.at(by,inv,a);draw=np.random.default_rng(8771).integers(len(tn),size=(1500,len(tn)))
    sampled=by[draw].sum(1);rr=np.sqrt(sampled[...,:4]/sampled[...,4:]);ci=np.quantile(rr[...,0:1]-rr,[.025,.975],axis=0)
    detail={str(sz):dict(rmse=r[j].tolist(),gain=(r[j,:,0:1]-r[j]).tolist(),gain95=ci[:,j].tolist(),
                  test_pixels_using_nonzero_shift=(mv[j,:,0]/mv[j,:,1]).tolist()) for j,sz in enumerate(SIZES)}
    named={}
    for b,name in enumerate(('415','5775')):
        named[name]={}
        for j,z in enumerate(['all','clear','partial','blocked']):
            vals={}
            for k,t in enumerate((0,5,10)):
                n,s,o,num=total[b,j,k];vals[str(t)]=dict(n=int(n),baseline_rmse=np.sqrt(s/n),oracle_rmse=np.sqrt(o/n),
                                            fraction_with_neighbor_within_1dB=num/n)
            named[name][z]=vals
    write('shift.json',dict(scope='Interior 252x252 only, both folds evaluated. One set of true labels estimates per-map or per-tile shifts/offsets, the other checkerboard parity tests. This is a diagnostic, unavailable at inference. Per-pixel oracle uses its own target and is more optimistic.',
         fit_offset='additive constant to predicted PL; at least 4 valid fit-parity pixels per tile, else identity',offsets=SHIFTS,
         variants=['no correction','shift','bias','shift+bias'],valid=u[0,:,4].astype(int).tolist(),sizes=detail,per_pixel_oracle=named))
    print('shift',[(size,np.round(r[j],4).tolist()) for j,size in enumerate(SIZES)],flush=True)
if __name__=='__main__':
    if sys.argv[1]=='finish':finish(int(sys.argv[2]))
    else:run(int(sys.argv[1]),int(sys.argv[2]))
