"""Use precisely aligned and wrong launch lattices as matched input-only exemplar probes."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cc_core import gpu_guard,atomic_json
from src.c93_retrieve import A,features,transfer,opoint,ap
def fields(s,idx):
    v=np.load(A/(s+'_features.npz'));phase=np.load(A/(s+'_phase.npz'))
    assert np.array_equal(phase['ids'],v['ids']) and np.array_equal(phase['pixels'],v['pixels'])
    x,st,fs=features(v,'flow')
    phase=phase['data'][:,:,idx,:].copy();
    phase[:,:,2:4]=phase[:,:,2:4].clip(-15,30)/10
    # Retain the same coverage/size statistic in all arms; only the launch phase is altered.
    return v['ids'],np.concatenate([x,phase],-1),st,fs
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();gpu=gpu_guard()
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(3);idx=a.index
    ids,x,st,fs=fields('screen',idx);s=np.load(A/'screen_target.npz');assert np.array_equal(s['ids'],ids)
    T={k:s[k] for k in ('weak','valid','y')};fit=s['groups']=='fit';sel=s['groups']=='select';out=A/f'phase{idx}';out.mkdir(exist_ok=True)
    order=np.flatnonzero(fit);np.random.default_rng(9391).shuffle(order);oof=np.full((len(ids),512,2,3),np.nan,np.float32)
    for part in np.array_split(order,5):
        q=np.zeros(len(ids),bool);q[part]=1;oof[q]=transfer(x,T,st,fit&~q,x[q],st[q],fs[q],fs)
    th=[opoint(T['weak'][fit,:,b][T['valid'][fit,:,b]],oof[fit,:,b,0][T['valid'][fit,:,b]]) for b in (0,1)]
    atomic_json(out/'choice.json',dict(angle=[0,.013,-.017][idx],threshold=th,gpu=gpu,selection='azimuth fixed a priori, F1 from 5 terrain folds'))
    np.savez_compressed(out/'oof.npz',ids=ids[fit],pred=oof[fit])
    z=transfer(x,T,st,fit,x[sel],st[sel],fs[sel],fs);np.savez_compressed(out/'select.npz',ids=ids[sel],pred=z)
    for stage in ('confirm','patch'):
        qi,qx,qs,qf=fields(stage,idx);pred=transfer(x,T,st,fit,qx,qs,qf,fs)
        np.savez_compressed(out/(stage+'.npz'),ids=qi,pred=pred)
    print('complete',idx,th,flush=True)
if __name__=='__main__':main()
