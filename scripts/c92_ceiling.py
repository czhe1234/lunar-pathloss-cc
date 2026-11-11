"""How much of frozen C82 error could the measured extra diffuse power remove even with perfect local routing?"""
import argparse,json
import numpy as np
from c92_score import prep,A
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','confirm']);stage=ap.parse_args().stage
    rec,raw,p,single,new,powers=prep(stage);q=np.array([e['group']==('select' if stage=='screen' else 'confirm') for e in rec])
    result={}
    for b in (0,1):
        y=raw[q,:,b];w=raw[q,:,4+b]>.5;pr=p[q,:,b];v=raw[q,:,6]
        increment=np.minimum(powers[-1][q,:,b]-powers[0][q,:,b],0)
        to_prediction=powers[-1][q,:,b]-pr
        error=y-pr
        def correct(d):return np.maximum(np.minimum(error,np.maximum(d,0)),np.minimum(d,0))
        names={'all':w,'shadow':w&(v<.01),'strong_extra_shadow':w&(v<.01)&(increment<=-3)}
        def stat(s):
            if not s.any():return dict(n=0,fraction_of_band=0.)
            base=float(np.sqrt(np.square(error[s].astype('float64')).mean()))
            prop=float(np.sqrt(np.square(error[s]-correct(increment)[s]).mean()))
            mix=float(np.sqrt(np.square(error[s]-correct(to_prediction)[s]).mean()))
            slope=float(np.mean(increment[s]*error[s])/np.maximum(1e-12,np.mean(increment[s]**2)))
            cor=np.corrcoef(increment[s],error[s])[0,1] if np.std(increment[s])>1e-8 else 0.
            return dict(n=int(s.sum()),fraction_of_band=s.sum()/w.sum(),baseline_rmse=base,oracle_add_RMSE=prop,oracle_add_gain=base-prop,
                oracle_mix_RMSE=mix,oracle_mix_gain=base-mix,best_one_scalar_on_same_data=slope,correlation_with_C82_residual=float(cor),
                fraction_needing_more_received_power=float(np.mean(error[s]<0)),mean_C82_error_PL=float(np.mean(-error[s],dtype='float64')))
        result[b]={key:stat(s) for key,s in names.items()}
    (A/(stage+'_ceiling.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({b:(v['all']['oracle_add_gain'],v['shadow']['correlation_with_C82_residual']) for b,v in result.items()})
if __name__=='__main__':main()
