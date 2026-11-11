"""Materialize paired TRAIN models, independent input packages, and joint-weak cohorts at immutable pixel IDs."""
import argparse,csv,gzip,json,sys
from pathlib import Path
import numpy as np
from c92_score import prep
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss';D=R.parent/'data'
def save_json(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def join(stage):
    rec,raw,p,core,multi,pl=prep(stage);phys=pl[-1].astype('float32')
    original=np.load(R/'audit/c78_clear/train_panel.npz');points=original['pixels'];o_ids=original['ids']
    ids=np.array([e['index'] for e in rec]);pos=np.array([e['pos'] for e in rec]);assert np.array_equal(o_ids[pos],ids)
    loc={}
    for d in sorted((R/'audit/c91_image').glob(stage+'[0-9]')):
        v=np.load(d/'loc.npy')
        for j,e in enumerate(json.load(open(d/'done.json'))['rows']):loc[e['pos']]=v[j]
    load=lambda k,fn:{int(i):(ar,j) for folder in sorted((R/'audit/c78_clear').glob('train_'+k+'_*of3'))
                       for ar in [np.load(folder/fn,mmap_mode='r')] for j,i in enumerate(np.load(folder/'positions.npy'))}
    geom=load('geom','geometry.npy');sheet=load('sheet','sheet.npy');v0=raw[:,:,6]
    geom=np.stack([geom[k][0][geom[k][1],loc[k]] for k in pos]).astype('float32')
    sheet=np.stack([sheet[k][0][sheet[k][1],loc[k]] for k in pos]).astype('float32')
    px=np.stack([points[k,loc[k]].astype('int32') for k in pos])
    y=raw[:,:,:2];w=raw[:,:,4:6].astype(bool);state=np.where(v0>.99,0,np.where(v0>=.01,1,2)).astype('uint8')
    rows=list(csv.DictReader(open(D/'train_index.csv')))
    labels=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')]
    masks=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
    bands=np.array([415e6,5.775e9]);fs=20*np.log10(geom[:,:,17:18].clip(.001)*bands*4*np.pi/299792458.)
    dmin=y-np.maximum(p,phys);weak=(dmin>=5)&w
    strict=(state[:,:,None]==0)&(y>fs+3)&(p<=fs+3)&(phys<=fs+3)&w
    delta=np.full(y.shape,np.nan,np.float32);local_ok=np.zeros(y.shape,bool)
    offsets=np.array([(a,b) for a in range(-4,5) for b in range(-4,5) if 4<=a*a+b*b<=16])
    for j,i in enumerate(ids):
        yy,xx=np.divmod(px[j],256);ry=yy[:,None]+offsets[:,0];cx=xx[:,None]+offsets[:,1]
        inside=(ry>=0)&(ry<256)&(cx>=0)&(cx<256);query=ry.clip(0,255)*256+cx.clip(0,255)
        assert len(np.unique(px[j]))==512 and rows[i]['terrain_id']==rec[j]['terrain']
        for b in range(2):
            yy2=labels[b][i].ravel();mm=np.unpackbits(masks[b][i]);assert np.array_equal(y[j,:,b],yy2[px[j]]) and np.array_equal(w[j,:,b],mm[px[j]])
            mw=inside & mm[query].astype(bool);n=mw.sum(1);sel=(n>=12)&w[j,:,b];local_ok[j,:,b]=sel
            kk=np.where(mw[sel],yy2[query[sel]].astype('float32'),np.nan)
            delta[j,sel,b]=y[j,sel,b]-np.nanmedian(kk,axis=1)
    names=np.array([e['terrain'] for e in rec]);groups=np.array([e['group'] for e in rec])
    np.savez_compressed(A/(stage+'_target.npz'),ids=ids,pos=pos,pixels=px,terrain=names,groups=groups,
                        y=y,valid=w,c82=p,c92=phys,weak=weak,strict=strict,annulus_delta=delta,annulus_valid=local_ok)
    # This file is input-only. No field derived from a released power or mask, no baseline predictions.
    np.savez_compressed(A/(stage+'_geo.npz'),ids=ids,pixels=px,geo=geom,sheet=sheet,visible=v0,state=state)
    if stage=='confirm':
        with gzip.open(A/'confirm_cases.csv.gz','wt',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['terrain','sample','tx_row','tx_col','rx_row','rx_col','band_MHz','state','truth_PL_dB','C82_PL_dB','C92_PL_dB','weaker_than_both_dB','local_dip_dB','strict_clear_miss'])
            for j,i in enumerate(ids):
                for l,b in np.argwhere(weak[j]):
                    writer.writerow([names[j],rows[i]['sample_id'],rows[i]['tx_row'],rows[i]['tx_col'],*divmod(int(px[j,l]),256),
                                     ('415','5775')[b],('clear','partial','shadow')[state[j,l]],round(float(y[j,l,b]),3),round(float(p[j,l,b]),3),
                                     round(float(phys[j,l,b]),3),round(float(dmin[j,l,b]),3),round(float(delta[j,l,b]),3) if local_ok[j,l,b] else '',int(strict[j,l,b])])
    summary={}
    for group in set(groups):
        g=groups==group;out=[]
        for b in range(2):
            t=y[g,:,b];pred=p[g,:,b];sq=(pred.astype('float64')-t)**2;ok=w[g,:,b];ww=weak[g,:,b]
            stats=[]
            for s in (-1,0,1,2):
                mask=ok&((state[g]==s) if s>=0 else True);e=ww&mask
                stats.append(dict(n=int(mask.sum()),miss=int(e.sum()),prevalence=float(e.sum()/mask.sum()),
                    sse_share=float(sq[e].sum()/sq[mask].sum()),c82_miss_rmse=float(np.sqrt(sq[e].mean())) if e.any() else None))
            test=lambda zz:float(sq[zz].sum()/sq[ok].sum())
            res={'by_state':stats,'sens':{str(t):int(((dmin[g,:,b]>=t)&ok).sum()) for t in (3,5,8)},
              'local_evaluable_miss':int((ww&local_ok[g,:,b]).sum()),'local_dip_miss':int((ww&(delta[g,:,b]>=3)).sum()),
              'local_dip_share_sse':test(ww&(delta[g,:,b]>=3)),
              'strict_clear':int(strict[g,:,b].sum()),'strict_severe_overlap':int((ww&strict[g,:,b]).sum()),
              'base_rmse':float(np.sqrt(sq[ok].mean())),'oracle_rmse_if_shared_perfect':float(np.sqrt(sq[ok&~ww].sum()/ok.sum())),
              'opposite_both_too_weak_5dB':int(((np.minimum(p[g,:,b],phys[g,:,b])-t>=5)&ok).sum())}
            out.append(res)
        ww=w[g].all(-1);both=weak[g].all(-1)
        summary[group]=dict(terrains=int(g.sum()),bands=out,dual_valid=int(ww.sum()),both_missed=int(both.sum()),either_joint_valid=int((weak[g].any(-1)&ww).sum()),
                            shared_rows=[int((weak[g,:,b]&ww).sum()) for b in (0,1)])
    save_json(A/(stage+'_inventory.json'),summary)
    print(stage,[(k,[a['by_state'][0]['miss'] for a in d['bands']]) for k,d in summary.items()])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);join(ap.parse_args().stage)
