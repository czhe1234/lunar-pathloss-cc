"""Paired controls for direct ray deposition on actual TRAIN labels; no inference-time label features."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver';C=R/'audit/c78_clear'
def db(x):return (-10*np.log10(x.clip(1e-5))).clip(-35,50)
def auc_u(s,w):
    pos=w.sum();neg=len(w)-pos
    if min(pos,neg)<1:return 0.,0.
    return rankdata(s)[w].sum()-pos*(pos+1)/2,pos*neg
def score(records,key,band):
    # Score all correction ranks, then compare within the same terrain/TX and count/distance bins.
    cc=np.concatenate([r[key] for r in records]);valid=cc[:,3+band]>.5;C0=cc[valid];weak=C0[:,5+band]>0
    raw=[]
    for j in (0,1):
        nu,de=auc_u(C0[:,j],weak);raw.append(nu/de if de else None)
    per=[];supported=0
    for r in records:
        x=r[key];q=x[:,3+band]>.5;w=x[:,5+band]>.5
        strata=x[:,2].astype(int);z=np.zeros(3)
        for i in np.unique(strata[q]):
            mask=q&(strata==i);u,n=auc_u(x[mask,0],w[mask]);v,k=auc_u(x[mask,1],w[mask]);assert k==n
            z+=[u,v,n]
        per.append(z);supported+=bool(z[2])
    per=np.stack(per);s=per.sum(0);rng=np.random.default_rng(86211)
    boot=per[rng.integers(len(per),size=(4000,len(per)))].sum(1);ok=boot[:,2]>0
    delta=(boot[ok,0]-boot[ok,1])/boot[ok,2]
    return dict(valid_pixels=len(C0),observed_weak=int(weak.sum()),raw_auc=raw,
                matched_auc_exact=float(s[0]/s[2]) if s[2] else None,matched_auc_shift=float(s[1]/s[2]) if s[2] else None,
                matched_exact_minus_shift_95=np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
                comparable_pos_neg_pairs=int(s[2]),terrains_with_pairs=int(supported),
                exact_fraction_direct_below_minus3=float((C0[:,0]>3).mean()),
                shifted_fraction_direct_below_minus3=float((C0[:,1]>3).mean()),
                predicted_direct_loss_IQR=np.quantile(C0[:,0],[.25,.5,.75]).tolist())
def main():
    plan=json.load(open(A/'c86_protocol.json'))['maps'];paths=sorted(A.glob('targeted*/*.npz'))
    assert len(paths)==len(plan) and {p.stem for p in paths}=={r['terrain'] for r in plan}
    ref=np.load(C/'train_reference.npy',mmap_mode='r')
    gs={}
    for d in C.glob('train_geom_*of*'):
        geo=np.load(d/'geometry.npy',mmap_mode='r')
        gs.update({j:(geo,k) for k,j in enumerate(np.load(d/'positions.npy'))})
    records=[];checks=[];cov=[];pads=[];hit_pad=[];direction=[];terms=0
    for p in paths:
        f=np.load(p);j=int(f['pos']);loc=f['loc'];raw=ref[j,loc];z,k=gs[j];geom=z[k,loc]
        true=np.load(A/f'rt_confirm0/{j}.npz')['features'][0,loc];v=f['values'];terms+=len(v[0])
        checks.extend(db(v[0,:,0])-true[:,0]);cov.extend(v[0,:,1]-true[:,1])
        if len(v)==5:pads.extend(db(v[2,:,0])-db(v[4,:,0]));hit_pad.extend(v[2,:,1]-v[4,:,1])
        cols=[];valid=raw[:,4:6];freq=np.array([415e6,5.775e9])
        weak=raw[:,:2] > 20*np.log10(4*np.pi/299792458*geom[:,17:18]*freq)+3
        bin0=np.digitize(geom[:,20],[20,100,400]);dbin=np.digitize(geom[:,18],[120])
        strata=bin0*2+dbin;feat={}
        for name,i in [('4M',0),('4B',2)]:
            feat[name]=np.column_stack([db(v[i,:,0]),db(v[i+1,:,0]),strata,valid,weak])
        records.append(feat);direction.append(f['directions'][:4])
    delta=np.array(checks);extra=np.array(pads)
    out=dict(terrains=len(paths),locations=terms,modes=['correct azimuth, same nominal N','+0.013 rad azimuth, same N, H, TX and measurement operator'],
        eligibility='risk-stratified candidate widely certified locations >=65 m, target subsets only; not an official score',
        full_4M_vs_sparse4M=dict(n=len(delta),rmse_dB=float(np.sqrt(np.mean(delta**2))),max_abs_dB=float(max(abs(delta))),max_hit_fraction_delta=float(max(abs(np.asarray(cov))))),
        sparse4B_pad_065_vs_135=dict(n=len(extra),rmse_dB=float(np.sqrt(np.mean(extra**2))),max_abs_dB=float(max(abs(extra))),max_hit_fraction_delta=float(max(abs(np.asarray(hit_pad))))),
        rays_per_map_median=np.median(np.stack(direction),axis=0).tolist(),bands={str(f):{k:score(records,k,b) for k in ('4M','4B')} for b,f in enumerate((415,5775))})
    (A/'targeted_sionna.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda v:v.item())+'\n')
    print(out['terrains'],terms,out['full_4M_vs_sparse4M'],out['sparse4B_pad_065_vs_135'],flush=True)
    for k,b in out['bands'].items():print(k,[(a,v['valid_pixels'],v['observed_weak'],v['matched_auc_exact'],v['matched_auc_shift'],v['matched_exact_minus_shift_95']) for a,v in b.items()],flush=True)
if __name__=='__main__':main()
