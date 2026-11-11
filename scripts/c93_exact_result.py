"""Paired original-solver direct-power ablation on untouched, contiguous RX windows."""
from pathlib import Path
import numpy as np,json
from c93_score import A,ap,ratio
def APmap(y,s,n,boot):
    order=np.argsort(-s,kind='stable');a=y[order].astype('float64');idx=n[order];p=s[order]
    ends=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1];vals=[]
    for counts in np.array_split(boot,30):
        w=counts[:,idx];z=np.cumsum(a*w,axis=1)[:,ends];nn=np.cumsum(w,axis=1)[:,ends]
        v=(np.diff(z,axis=1,prepend=np.zeros((len(z),1)))*z/np.maximum(nn,1)).sum(1)/np.maximum(z[:,-1],1);vals.append(v)
    return np.concatenate(vals)
def main():
    plan=json.load(open(A/'exact_windows.json'))['maps'];t=np.load(A/'patch_target.npz');feature=np.load(A/'patch_features.npz')
    ids=t['ids'];px=t['pixels'];w=t['valid'];y=t['y'];b0=t['c82'];phy=t['c92'];weak=t['weak'];fs=feature['fs']
    st=feature['x'][:,:,feature['names'].tolist().index('state')];source={};padding=[];nn=[]
    for path in sorted(A.glob('exact[0-9]')):
        meta=json.load(open(path/'done.json'))
        for p in path.glob('*.npz'):
            f=np.load(p);i=int(f['index']);v=f['values'];source[i]=(v[:3],f['pixel'])
            if len(v)>3:padding.append((v[3]-v[0],v[0]))
            nn.extend(f['directions'][:3].tolist())
    assert set(source)=={e['index'] for e in plan}
    obs=[]
    for it,e in enumerate(plan):
        j=e['patch_map'];ix=e['windows'];i=e['index'];dat,qq=source[i]
        assert np.array_equal(px[j].reshape(16,49)[ix].ravel(),qq) and ids[j]==i
        for jj,h in enumerate(ix):
            sl=slice(h*49,h*49+49)
            obs.append([it,h,e['arms'][jj],y[j,sl],w[j,sl],weak[j,sl],b0[j,sl],phy[j,sl],fs[j,sl],st[j,sl],dat[:,jj*49:(jj+1)*49]])
    group=np.array([o[0] for o in obs]);arm=np.array([o[2] for o in obs])
    y,w,evt,base,phy,fs,v=np.stack([o[3] for o in obs]),np.stack([o[4] for o in obs]),np.stack([o[5] for o in obs]),np.stack([o[6] for o in obs]),np.stack([o[7] for o in obs]),np.stack([o[8] for o in obs]),np.stack([o[9] for o in obs])
    reads=np.stack([o[10] for o in obs]);out={};boot=np.random.default_rng(934229).multinomial(len(plan),np.full(len(plan),1/len(plan)),size=1000)
    for b,frequency in enumerate((415e6,5.775e9)):
        row={}
        for select in ('all','uniform','grazing'):
            sel=np.ones(len(arm),bool) if select=='all' else arm==select
            valid=w[:,:,b]&((v==0)&sel[:,None]);case=evt[:,:,b]&valid;true=y[:,:,b];mp=np.broadcast_to(group[:,None],valid.shape)[valid]
            # Direct-only on the same physical plane, scale with free-space frequency. On clear cells the old normalized direct is ~1.
            # Keeping C92 input-only total as the extra-power reference introduces a small registration approximation and a floor.
            extra=np.maximum(10**((fs[:,:,b].astype('float64')-phy[:,:,b])/10)-1,0)
            score=[];bs=[]
            for j in range(3):
                direct_norm=reads[:,j,:,0];risk=-10*np.log10(direct_norm.clip(1e-12))
                directPL=20*np.log10(4*np.pi*frequency/299792458)-10*np.log10(reads[:,j,:,2].clip(1e-20))
                alert=risk>=3;tp=alert&case;alone=directPL[case];total=fs[:,:,b]-10*np.log10((direct_norm+extra).clip(1e-13))
                mse=lambda z:float(np.sqrt(np.mean((z[valid].astype('float64')-true[valid])**2)))
                a=risk.reshape(-1,7,7);e=case.reshape(-1,7,7);validxy=valid.reshape(-1,7,7);num=win=0
                for dx in (0,1):
                    va=a[:,:,:-1] if dx==0 else a[:,:-1,:];vb=a[:,:,1:] if dx==0 else a[:,1:,:]
                    ea=e[:,:,:-1] if dx==0 else e[:,:-1,:];eb=e[:,:,1:] if dx==0 else e[:,1:,:]
                    ma=validxy[:,:,:-1] if dx==0 else validxy[:,:-1,:];mb=validxy[:,:,1:] if dx==0 else validxy[:,1:,:]
                    k=ma&mb&(ea!=eb);delta=np.where(ea,va-vb,vb-va);win+=((np.sign(delta[k])+1)*.5).sum();num+=k.sum()
                stats=dict(AP=ap(case[valid],risk[valid]),precision=ratio(tp.sum(),(alert&valid).sum()),recall=ratio(tp.sum(),case.sum()),alerts=int((alert&valid).sum()),
                    uncovered_weight_mean_on_cases=float((1-reads[:,j,:,1])[case].mean()),direct_still_brighter_than_true_by3dB=ratio((alone<true[case]-3).sum(),case.sum()),
                    standalone_direct_plus_extra_RMSE=mse(total),C92_reference_RMSE=mse(phy[:,:,b]),C82_reference_RMSE=mse(base[:,:,b]),
                    adjacent_pairs=int(num),adjacent_order=ratio(win,num))
                score.append(stats);bs.append(APmap(case[valid],risk[valid],mp,boot))
            row[select]=dict(windows=int(sel.sum()),valid=int(valid.sum()),events=int(case.sum()),prevalence=ratio(case.sum(),valid.sum()),
                nominal_vs_mean_wrong_AP=float(score[0]['AP']-.5*(score[1]['AP']+score[2]['AP'])),
                nominal_vs_mean_wrong_AP95=np.quantile(bs[0]-.5*(bs[1]+bs[2]),[.025,.975]).tolist(),phases=score)
        out[b]=row
    differences=np.concatenate([v[0] for v in padding])
    qc=dict(padded_maps=len(padding),tested_points=len(differences),max_abs_gain_change=float(abs(differences[:,0]).max()),mean_abs_gain_change=float(abs(differences[:,0]).mean()),
        max_hit_fraction_change=float(abs(differences[:,1]).max()),median_selected_launch_per_scene_phase=float(np.median(nn)))
    (A/'exact_result.json').write_text(json.dumps(dict(terrains=len(plan),windows=len(arm),qa=qc,bands=out),indent=2,allow_nan=False)+'\n')
    print('done',qc,[(b,out[b]['all']['events'],out[b]['all']['nominal_vs_mean_wrong_AP'],out[b]['all']['nominal_vs_mean_wrong_AP95']) for b in range(2)])
if __name__=='__main__':main()
