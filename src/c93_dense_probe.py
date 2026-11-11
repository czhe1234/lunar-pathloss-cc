"""Repeat the frozen local lookup with 8x calibration density and no new terrains or learned weights."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import atomic_json,gpu_guard
from src.c93_retrieve import A,features,transfer,opoint
def power_cutoff(score,amp,y,p,valid):
    risk=score[valid];base=p[valid].astype('float64');truth=y[valid].astype('float64');a=amp[valid].astype('float64')
    order=np.argsort(-risk,kind='stable');j=np.r_[np.flatnonzero(risk[order][1:]!=risk[order][:-1]),len(risk)-1]
    benefit=np.cumsum(((truth-base)**2-(truth-a)**2)[order])[j]
    k=np.r_[0.,benefit].argmax()
    return float(risk[order[j[k-1]]]) if k else 2.
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['screen','confirm','patch'],required=True);a=ap.parse_args()
    gpu=gpu_guard();torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False
    p=np.load(A/'ref_plan.npz');ids=p['ids'];ref={}
    for f in sorted(A.glob('ref_[0-9]')):
        q=np.load(f/'examples.npz');fields=q['x'];tar=q['targets'];ff=q['fs'];names=q['names'];px=q['pixels']
        for j,i in enumerate(q['ids']):ref[i]=(fields[j],tar[j],ff[j],px[j])
    assert len(ref)==80
    v=np.stack([ref[i][0] for i in ids]);tar=np.stack([ref[i][1] for i in ids]);fs=np.stack([ref[i][2] for i in ids])
    qx={'names':names,'x':v,'fs':fs};x,st,fs=features(qx,'flow')
    T={'y':tar[:,:,:,0],'valid':tar[:,:,:,1]>.5,'weak':(tar[:,:,:,0]-np.maximum(tar[:,:,:,2],tar[:,:,:,3])>=5)&(tar[:,:,:,1]>.5)}
    old=np.load(A/'screen_target.npz');oldi=old['groups']=='fit';assert np.array_equal(old['ids'][oldi],ids)
    diff=np.abs(old['c92'][oldi]-tar[:,:512,:,3])
    changed=int(np.count_nonzero(T['weak'][:,:512]!=old['weak'][oldi]));assert np.quantile(diff,.99)<.05 and changed<15,(diff.max(),changed)
    fit=np.ones(len(ids),bool);out=A/'dense_probe';out.mkdir(exist_ok=True);ts=time.monotonic()
    if a.stage=='screen':
        order=np.arange(len(ids));np.random.default_rng(9391).shuffle(order)
        oof=np.empty((len(ids),512,2,3),np.float32)
        for part in np.array_split(order,5):
            q=np.zeros(len(ids),bool);q[part]=True
            oof[q]=transfer(x,T,st,~q,x[q,:512],st[q,:512],fs[q,:512],fs)
        threshold=[];mse=[]
        for b in range(2):
            w=T['valid'][:,:512,b];y=T['weak'][:,:512,b][w];threshold.append(opoint(y,oof[:,:,b,0][w]))
            mse.append(power_cutoff(oof[:,:,b,0],oof[:,:,b,2],T['y'][:,:512,b],tar[:,:512,b,2],w))
        np.savez_compressed(out/'oof.npz',ids=ids,pred=oof)
        atomic_json(out/'choice.json',dict(threshold=threshold,power_threshold=mse,replay_max_dB=diff.max((0,1)).tolist(),cohort_changes=changed,
          points=int(T['weak'].shape[0]*T['weak'].shape[1]),valid=T['valid'].sum((0,1)).tolist(),weak=T['weak'].sum((0,1)).tolist(),
          gpu=gpu,rule='same C93 flow features, fixed k and equal terrain cap; operating points set on five out-of-terrain folds'))
        v=np.load(A/'screen_features.npz');qq,qs,qf=features(v,'flow');q=old['groups']=='select';qi=v['ids'][q];qq,qs,qf=qq[q],qs[q],qf[q];name='select'
    else:
        assert (out/'choice.json').is_file()
        v=np.load(A/(a.stage+'_features.npz'));qq,qs,qf=features(v,'flow');qi=v['ids'];name=a.stage
    pred=transfer(x,T,st,fit,qq,qs,qf,fs)
    np.savez_compressed(out/(name+'.npz'),ids=qi,pred=pred);print(name,len(qi),'seconds',time.monotonic()-ts,flush=True)
if __name__=='__main__':main()
