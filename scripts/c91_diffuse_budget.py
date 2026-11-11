"""On the same held-out positions compare measured total with first-order power; classify both signs."""
from pathlib import Path
import json,numpy as np
from c91_score import load,R,A
def main():
    rec,rr,p,branch,image=load('confirm')
    src={}
    for d in sorted((R/'audit/c78_clear').glob('train_power_*of3')):
        ar=np.load(d/'power.npy',mmap_mode='r');src.update({pos:(ar,j) for j,pos in enumerate(np.load(d/'positions.npy'))})
    locs={}
    for d in sorted(A.glob('confirm[0-9]')):
        loc=np.load(d/'loc.npy')
        for j,r in enumerate(json.load(open(d/'done.json'))['rows']):locs[r['pos']]=loc[j]
    V=np.stack([src[e['pos']][0][src[e['pos']][1],locs[e['pos']]] for e in rec])
    out={}
    for b,f in enumerate([415e6,5.775e9]):
        w=(rr[:,:,4+b]>.5)&(rr[:,:,6]<.01);s=(p[:,:,b]-rr[:,:,b]).astype('float64')**2
        r=np.expm1(branch[:,:,4].astype('float64')).clip(.25);fs=20*np.log10(r*4*np.pi*f/299792458)
        direct=10**(-branch[:,:,6+5*b].astype('float64')/10);spec=10**(-image[:,:,5+b].astype('float64')/10)
        d=V[:,:,14+b].astype('float64')*r**2
        n=V[:,:,12+b].astype('float64')*r**2
        power=direct+spec+d;ex=fs-10*np.log10(np.maximum(power,1e-17))-rr[:,:,b]
        a={}
        for tag,subset in [('predicted_power_at_least_10dB_too_low',ex>=10),('within_3dB',abs(ex)<=3),('predicted_power_at_least_10dB_too_high',ex<=-10)]:
            valid=w&subset;a[tag]={'pixel_fraction':float(valid.sum()/w.sum()),'C82_shadow_SSE_fraction':float(s[valid].sum()/s[w].sum())}
        a['n']=int(w.sum());a['component_candidate_RMSE']=float(np.sqrt((ex[w]**2).mean()))
        a['candidate_pred_minus_true_PL_quantiles']=np.quantile(ex[w],[.1,.5,.9]).tolist()
        a['fraction_diffuse_larger_than_straight_plus_specular']=float((d[w]>direct[w]+spec[w]).mean())
        a['spherical_vs_MIS_diffuse_median_difference_dB']=float(np.median(abs(10*np.log10((d[w]+1e-15)/(n[w]+1e-15)))))
        out[str(b)]=a
    (A/'diffuse_budget.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
