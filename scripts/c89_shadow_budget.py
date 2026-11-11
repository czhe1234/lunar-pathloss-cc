"""Compare the sole straight-path energy with measured total energy on held-out TRAIN."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c89_survival';C=R/'audit/c88_mesh'
def main():
    raw=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r');px=np.load(R/'audit/c78_clear/train_panel.npz')['pixels']
    base=np.load(C/'base.npy',mmap_mode='r');g=np.array(json.load(open(C/'base_index.json'))['groups'])
    info=np.zeros((2,2,4));joint=np.zeros((2,4));maps=0
    for s in sorted(A.glob('shard*')):
        f=np.load(s/'features.npy',mmap_mode='r');ix=json.load(open(s/'done.json'))['positions']
        for j,k in enumerate(ix):
            if g[k]!='confirm':continue
            maps+=1;v=f[j].astype(np.float64);rr=raw[k];p=base[k].reshape(2,-1)[:,px[k]]
            for b,freq in enumerate([415e6,5.775e9]):
                fs=20*np.log10((4*np.pi*freq/299792458)*np.expm1(v[:,4]).clip(.25))
                direct=fs+v[:,6+5*b];diff=direct-rr[:,b]
                mask=(rr[:,4+b]>.5)&(rr[:,6]<.01);se=(p[b]-rr[:,b])**2
                for col,sel in enumerate([diff>=10,(v[:,17+12*b]<.05)]):
                    a=info[b,col];a[0]+=mask.sum();a[1]+=(sel&mask).sum();a[2]+=se[mask].sum(dtype=np.float64);a[3]+=se[sel&mask].sum(dtype=np.float64)
                both=(rr[:,4]>.5)&(rr[:,5]>.5)&(rr[:,6]<.01);a=joint[b]
                a[0]+=both.sum();a[1]+=(diff[both]>=10).sum();a[2]+=np.clip(diff[both],-50,110).sum()
    out=dict(scope='484 TRAIN confirmation terrains, 1934 maps, 2048 prespecified pixels per map. Fully shadowed by frozen C82 registered geometry. Approximation, not a path-specific ground-truth measurement.',maps=maps)
    for b in range(2):
        out[['415','5775'][b]]={k:dict(valid=int(v[0]),fraction=float(v[1]/v[0]),fraction_of_shadow_C82_squared_error=float(v[3]/v[2])) for k,v in zip(['total_at_least_10dB_above_modeled_straight','modeled_per_seed_direct_hit_probability_under_5pct'],info[b])}
    (A/'shadow_budget.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
