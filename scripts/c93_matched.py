"""Can H and TX distinguish pairs for which both frozen models made nearly the same power prediction?"""
import json
import numpy as np
from c93_score import A
def main():
    q=np.load(A/'confirm_target.npz');valid=q['valid'];weak=q['weak'];base=q['c82'];phys=q['c92']
    f=np.load(A/'confirm_features.npz');p=np.load(A/'lookup_flow/confirm.npz')['pred']
    names=f['names'].tolist();x=f['x'];dist=np.expm1(x[:,:,names.index('range')]);state=np.rint(2*x[:,:,names.index('state')]).astype(int)
    out={}
    for b in (0,1):
        stats=np.zeros((len(x),5))
        for j in range(len(x)):
            w=valid[j,:,b];evt=weak[j,:,b];pos=np.flatnonzero(w&evt);neg=np.flatnonzero(w&~evt)
            if not len(pos) or not len(neg):continue
            i=pos[:,None];k=neg[None,:]
            keep=(state[j,i]==state[j,k])&(abs(dist[j,i]-dist[j,k])<=10)&(abs(base[j,i,b]-base[j,k,b])<=1.5)&(abs(phys[j,i,b]-phys[j,k,b])<=2)
            delta=p[j,i,b,0]-p[j,k,b,0];h=.5+.5*np.sign(delta)
            stat=stats[j];stat[0]=keep.sum();stat[1]=(h*keep).sum();stat[2]=keep.any(1).sum()
            a=keep.sum(1);m=a>0;stat[3]=((h*keep).sum(1)[m]/a[m]).sum();stat[4]=len(pos)
        c=stats.sum(0);rng=np.random.default_rng(93711);boot=stats[rng.integers(len(x),size=(2000,len(x)))].sum(1)
        out[b]=dict(terrains_w_pairs=int((stats[:,0]>0).sum()),pairs=int(c[0]),matched_misses=int(c[2]),total_misses=int(c[4]),
                   pair_weight_auc=c[1]/c[0],pair_auc95=np.quantile(boot[:,1]/boot[:,0],[.025,.975]).tolist(),equal_miss_auc=c[3]/c[2],
                   equal_miss_auc95=np.quantile(boot[:,3]/boot[:,2],[.025,.975]).tolist())
    (A/'matched_twins.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
