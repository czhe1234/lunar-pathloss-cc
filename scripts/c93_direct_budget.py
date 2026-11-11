"""Direct-only power lower bound and within-terrain dual-band concordance of common weak events."""
import json
import numpy as np
from c93_score import A
from c91_score import load as observations
def main():
    t=np.load(A/'confirm_target.npz');_,raw,p,br,im=observations('confirm')
    d=np.expm1(br[:,:,4].astype('float64')).clip(.25);fs=20*np.log10(d[:,:,None]*np.array([415e6,5.775e9])*4*np.pi/299792458)
    direct=fs+br[:,:,[6,11]].astype('float64');mirror=fs+im[:,:,5:7].astype('float64')
    w=t['valid'];y=t['y'];hit=t['weak'];e=y-direct
    state=np.where(raw[:,:,6]>.99,0,np.where(raw[:,:,6]>=.01,1,2));validboth=w.all(2)
    out={}
    for b in (0,1):
        out[b]={}
        for st,name in [(-1,'all'),(0,'clear'),(1,'partial'),(2,'shadow')]:
            m=hit[:,:,b]&((state==st) if st>=0 else True);cert=e[:,:,b]>=5.01
            sub=m&cert
            out[b][name]=dict(events=int(m.sum()),direct_alone_over_by_5dB=int(sub.sum()),fraction=float(sub.sum()/m.sum()),
               directly_certified_C82_event_SSE_share=float((((y[:,:,b]-p[:,:,b])**2)*sub).sum(dtype='float64')/(((y[:,:,b]-p[:,:,b])**2)*m).sum(dtype='float64')),
               required_direct_reduction_quantile10_median_p90_dB=np.quantile(e[:,:,b][sub],[.1,.5,.9]).tolist() if sub.any() else [])
    rawboth=hit.all(2);joint=[]
    for st,name in [(-1,'all'),(0,'clear'),(1,'partial'),(2,'shadow')]:
        jj=validboth&((state==st) if st>=0 else True)
        n=jj.sum(1);na=(hit[:,:,0]&jj).sum(1);nb=(hit[:,:,1]&jj).sum(1);ab=(rawboth&jj).sum(1)
        independent=na*nb/np.maximum(n,1)
        joint.append(dict(state=name,same_pixel_both=int(ab.sum()),expected_if_shuffled_within_map_and_state=float(independent.sum()),
               fraction_high_events_also_low=float(ab.sum()/nb.sum()),fraction_low_events_also_high=float(ab.sum()/na.sum())))
    out['frequency_shared']=joint
    (A/'direct_budget.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(out)
if __name__=='__main__':main()
