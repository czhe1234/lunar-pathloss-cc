"""Numerical seed stability and same-terrain TX counterfactual. No neighboring targets enter a prediction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from c93_geometry import read_compact
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss';D=R.parent/'data'
def load(prefix):
    dst={}
    for d in sorted(A.glob(prefix+'[0-9]')):
        arr=np.load(d/'power.npy');ids=json.load(open(d/'done.json'))['ids']
        dst.update({i:arr[j] for j,i in enumerate(ids)})
    return dst
def seed():
    t=np.load(A/'confirm_target.npz');risk=np.load(A/'dense_probe/confirm.npz')['pred']
    th=json.load(open(A/'dense_probe/choice.json'))['threshold'];x=load('confirm_seed');ids=t['ids'];new=np.stack([x[i] for i in ids])
    y=t['y'];orig=t['c92'];w=t['valid'];p=t['c82'];evt=t['weak']
    hits=(y[:,None]>=np.maximum(new,p[:,None])+5)&w[:,None]
    stable=evt&hits.all(1);report={}
    for b in (0,1):
        ww=w[:,:,b];m=evt[:,:,b];h=stable[:,:,b];k=risk[:,:,b,0]>=th[b]
        obs={}
        for tag,s in [('original',m),('stable_all_three',h)]:
            tp=ww&s&k;calc=risk[:,:,b,2];sub=s&ww
            obs[tag]=dict(count=int(sub.sum()),detected=int(tp.sum()),precision=float(tp.sum()/(k&ww).sum()),
                      recall=float(tp.sum()/sub.sum()),given_event_MAE=float(abs(calc-y[:,:,b])[sub].mean()),
                      recall_and_1db=float((tp&(abs(calc-y[:,:,b])<=1)).sum()/sub.sum()))
        delta=abs(new[:,:,:,b]-orig[:,None,:,b]);alts=[]
        for j in range(2):
            raw=hits[:,j,:,b];alts.append(dict(alternate_count=int(raw.sum()),retained_fraction=float((raw&m).sum()/m.sum()),
                 jaccard=float((raw&m).sum()/(raw|m).sum()),PL_difference_median_p95_p99=np.quantile(delta[:,j][ww],[.5,.95,.99]).tolist()))
        report[b]=dict(outcomes=obs,still_positive_both_new_seeds=float(h.sum()/m.sum()),alternates=alts)
    (A/'seed_stability.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(A/'seed_stable_flags.npz',ids=ids,pixels=t['pixels'],stable=stable,alternate_hits=hits)
    print([(b,r['still_positive_both_new_seeds']) for b,r in report.items()])
def paired(common=True):
    z=json.load(open(A/'tx_plan.json'));rows=list(csv.DictReader(open(D/'train_index.csv')));orig=load('tx');phys=load('tx_seed') if common else orig
    meta=json.load(open(R/'audit/c88_mesh/base_index.json'));pidx={i:j for j,i in enumerate(meta['ids'])};base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
    gt=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];mask=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
    cp=read_compact();H=np.load(D/'train_hm.npy',mmap_mode='r');table={}
    for i in z['ids']:
        if i not in phys:continue
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);px=np.array(z['pixels'][row['terrain_id']]);h=H[int(row['hm_row'])]
        y=np.stack([a[i].ravel()[px] for a in gt],1).astype('float32');w=np.stack([np.unpackbits(a[i])[px] for a in mask],1).astype(bool)
        p=base[pidx[i]].reshape(2,-1)[:,px].T;ph=(phys[i][0] if common else phys[i]);alt=orig[i];hit=y>=np.maximum(p,ph)+5;ohit=y>=np.maximum(p,alt)+5
        sy,sx=np.divmod(px,256);r=np.sqrt((sy-tr)**2+(sx-tc)**2+(h.ravel()[px]+1-(np.floor(h[min(255,tr+1),tc])+3))**2).clip(.25)
        path=20*np.log10(r);store,j=cp[i];vis=store[j,16].ravel()[px];state=np.where(vis>.99,0,np.where(vis>=.01,1,2))
        table[i]=(y,w,hit,y-path[:,None],state,ohit,ph)
    out={}
    for tag in ('identical','near','far'):
        allpairs=[p for p in z['pairs'] if p['group']==tag];pp=[p for p in allpairs if p['i'] in table and p['j'] in table]
        if not pp:
            out[tag]=dict(pairs=0,planned_pairs=len(allpairs));continue
        d=np.zeros((2,2,10));duplicate=None
        for p in pp:
            ya,wa,ha,ea,sa,oa,pha=table[p['i']];yb,wb,hb,eb,sb,ob,phb=table[p['j']]
            for b in (0,1):
                for subset in (0,1):
                    w=wa[:,b]&wb[:,b]&((sa==sb) if subset else True)
                    x=ha[:,b]&w;y=hb[:,b]&w
                    step=np.abs(ea[:,b]-eb[:,b])
                    d[b,subset]+=[w.sum(),x.sum()+y.sum(),2*(x&y).sum(),(x|y).sum(),step[x].sum()+step[y].sum(),
                        np.sum(step[w]),(x!=y).sum(),np.sum((oa[:,b]!=ha[:,b])&w)+np.sum((ob[:,b]!=hb[:,b])&w),
                        ((ya[:,b]-yb[:,b])[w].astype('float64')**2).sum(),np.sum(w&(step>5))]
            if tag=='identical':
                duplicate={'IDs':[rows[p['i']]['sample_id'],rows[p['j']]['sample_id']],
                           'same_fixed_stream_PHYS_max_difference_dB':np.max(abs(pha-phb),axis=0).tolist()}
        summary={}
        for b in (0,1):
            rr={}
            for subset,name in enumerate(('all_joint_valid','same_state')):
                n,s,both,e,ae,aa,flip,orig_flip,sq,over=d[b,subset]
                rr[name]=dict(joint_valid=int(n),positive_TX_oriented=int(s),positive_both=int(both//2),either_position=int(e),
                      positive_reappears_same_pixel=float(both/s) if s else None,
                      excess_PL_MAE_at_positive_one_end=float(ae/s) if s else None,excess_PL_MAE_overall=float(aa/n),
                      raw_label_pair_RMSE=float(np.sqrt(sq/n)),physical_seed_changes=int(orig_flip),event_change_positions=int(flip))
            summary[b]=rr
        out[tag]=dict(pairs=len(pp),planned_pairs=len(allpairs),terrains=len({p['terrain'] for p in pp}),separation_m=np.quantile([p['separation'] for p in pp],[0,.5,1]).tolist(),
                    bands=summary,exact_input_control=duplicate)
    (A/('tx_pair_summary_common.json' if common else 'tx_pair_summary_original.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print([(k,v['pairs'],[v['bands'][b]['all_joint_valid']['positive_reappears_same_pixel'] for b in (0,1)]) for k,v in out.items() if v['pairs']])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['seed','tx']);a=ap.parse_args()
    if a.mode=='seed':seed()
    else:
        paired(common=False)
        paired(common=True)
