"""Paired geometry ablations over the C82 residual, using only official TRAIN."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c88_mesh';DATA=R.parent/'data'
def arrays(stage):
    items=[];modes=None
    for f in sorted(A.glob(stage+'_*')):
        if not f.is_dir():continue
        meta=json.load(open(f/'done.json'));modes=meta['modes'];x=np.load(f/'features.npy',mmap_mode='r')
        items.extend([(e,x[j]) for j,e in enumerate(meta['records'])])
    items.sort(key=lambda t:t[0]['pos'])
    pos=np.array([e['pos'] for e,_ in items]);ids=np.array([e['index'] for e,_ in items])
    px=np.load(R/'audit/c78_clear/train_panel.npz')['pixels'][pos].astype(np.int64)
    raw=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r')[pos]
    info=json.load(open(A/'base_index.json'));assert np.array_equal(np.array(info['ids'])[pos],ids)
    pred=np.load(A/'base.npy',mmap_mode='r');p=np.stack([pred[j].reshape(2,-1)[:,s].T for j,s in zip(pos,px)])
    rows=list(csv.DictReader(open(DATA/'train_index.csv')));hh=np.load(DATA/'train_hm.npy',mmap_mode='r');fs=[]
    for i,pp in zip(ids,px):
        r=rows[i];h=hh[int(r['hm_row'])];x=int(r['tx_col']);y=int(r['tx_row']);pz=np.floor(h[min(y+1,255),x])+3
        dist=np.sqrt((pp%256-x)**2+(pp//256-y)**2+(h.ravel()[pp]+1-pz)**2).clip(.25)
        fs.append(20*np.log10(dist)*(1.)+20*np.log10(4*np.pi*415e6/299792458))
    return items,modes,raw,p,np.array(fs).astype('float32')
def shared(p,raw,fs,b):
    vis=raw[:,:,6];delta=(p[:,:,1]-p[:,:,0]-20*np.log10(5.775e9/415e6))/12
    extra=(p[:,:,b]-fs-(20*np.log10(5.775e9/415e6) if b else 0))/12
    return np.stack([np.ones_like(vis),vis,delta.clip(-2,5),extra.clip(-2,8),(fs-65)/15,1-vis,np.maximum(extra,0),np.maximum(delta,0)],-1)
def phys(v,raw,p,b):
    v=v.astype(np.float32);vis=raw[:,:,6:7];delta=((p[:,:,1:2]-p[:,:,0:1]-20*np.log10(5.775e9/415e6))/15).clip(0,3)
    z=v[:,:,[8,9,10,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30,31,32,35,36,37+b,39+b,41,42]].copy()
    z[:,:,[22,23]]*=1;z[:,:,24:26]/=25;z=np.clip(z,-8,8)
    return np.concatenate([z,z*vis,z*delta],-1)
def fit(x,t,w):
    k=x[w].astype('float64');y=t[w].astype('float64')
    gram=k.T@k;d=.012*np.diag(gram).clip(len(k)*.001);d[0]=.01
    return np.linalg.solve(gram+np.diag(d),k.T@y)
def evaluate(pred,raw,p):
    out=[]
    for b in range(2):
        m=raw[:,:,4+b]>.5;Y=raw[:,:,b];e=pred[:,:,b]-Y
        arr=[]
        for w in [m,m&(raw[:,:,6]>.99),m&(raw[:,:,6]<.01),m&(raw[:,:,6]>=.01)&(raw[:,:,6]<=.99)]:
            arr.append(np.stack([(e.astype('float64')**2*w).sum(1),w.sum(1)],1))
        out.append(np.stack(arr,1))
    return np.stack(out,1)
def rms(x):
    z=x.sum(0);return np.sqrt(z[...,0]/z[...,1])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);stage=ap.parse_args().stage
    rec,names,raw,p,fs=arrays(stage);gg=np.array([e['group'] for e,_ in rec]);fit_s=gg=='fit'
    score_s=gg==('select' if stage=='screen' else 'confirm')
    if stage=='screen':models={'shared':[],'nominal':['256_fixed']};models.update({k:['256_fixed',k] for k in names if k!='256_fixed'})
    else:
        choice=json.load(open(A/'selection.json'));models=choice['models']
    result={};coefs={};all_stats={};xmap={name:np.stack([v[names.index(name)] for e,v in rec]) for name in names}
    base=evaluate(p[score_s],raw[score_s],p[score_s]);base_rmse=rms(base)
    for label,variants in models.items():
        pr=p[score_s].copy();coefs[label]=[]
        for b in (0,1):
            blocks=[shared(p,raw,fs,b)];ref=None
            for v in variants:
                curr=phys(xmap[v],raw,p,b)
                blocks.append(curr if ref is None else curr-ref);ref=curr if ref is None else ref
            X=np.concatenate(blocks,-1)
            if stage=='screen':co=fit(X[fit_s],(raw[:,:,b]-p[:,:,b])[fit_s],(raw[:,:,4+b]>.5)[fit_s])
            else:co=np.array(choice['coefs'][label][b],dtype='float32')
            coefs[label].append(co.tolist())
            pr[:,:,b]+=(X[score_s]@co.astype('float32')).clip(-15,15)
        s=evaluate(pr,raw[score_s],p[score_s]);rm=rms(s);all_stats[label]=s
        result[label]={'rmse':rm.tolist(),'gain':(base_rmse-rm).tolist()}
    if stage=='screen':
        chosen=min((k for k in models if k not in ('shared','nominal','256_roll','256_shortroll')),
                   key=lambda k:np.array(result[k]['rmse'])[:,0].mean())
        control='256_shortroll' if chosen=='256_short' else '256_roll'
        keys=['shared','nominal',chosen,control]
        v={'selected':chosen,'physical_modes':sorted({k for l in keys for k in models[l]}),
           'models':{k:models[k] for k in keys},'coefs':{k:coefs[k] for k in keys}}
        (A/'selection.json').write_text(json.dumps(v)+'\n')
    else:
        np.savez(A/'confirm_physical_maps.npz',C82=base,**all_stats)
    out={'stage':stage,'terrains':int(score_s.sum()),'one_map_per_terrain':True,'sampled_points_before_mask':int(score_s.sum()*2048),
         'C82':base_rmse.tolist(),'columns':['all','clear','shadow','partial'],'scores':result}
    (A/(stage+'_physical.json')).write_text(json.dumps(out,indent=2)+'\n')
    print(stage,'C82',base_rmse[:,0], {k:np.round(np.array(v['gain'])[:,0],4).tolist() for k,v in result.items()},flush=True)
if __name__=='__main__':main()
