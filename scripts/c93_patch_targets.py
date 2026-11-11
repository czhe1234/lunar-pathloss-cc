"""Evaluate fixed adjacent-cell predictions after independent input-only patch selection."""
import csv,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss';D=R.parent/'data'
def main():
    plan=json.load(open(A/'patches.json'));f=np.load(A/'patch_features.npz')
    base=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r');dev=json.load(open(R/'audit/c88_mesh/base_index.json'))['ids']
    confirm=np.load(A/'confirm_target.npz');pidx={i:j for j,i in enumerate(confirm['ids'])}
    rows=list(csv.DictReader(open(D/'train_index.csv')));yy=[np.load(D/f'train_pl{k}.npy',mmap_mode='r') for k in ('415','58')];ww=[np.load(D/f'train_mask{k}.npy',mmap_mode='r') for k in ('415','58')]
    ps={}
    for k in sorted(A.glob('patch_power[0-9]')):
        v=np.load(k/'p.npy');ps.update({n:v[j] for j,n in enumerate(json.load(open(k/'done.json'))['ids'])})
    assert set(ps)==set(f['ids'])
    vals=[];truth=[];m=[];b=[];terrain=[];reg=[];diff=[];changed=[]
    for j,i in enumerate(f['ids']):
        px=f['pixels'][j];p=ps[i];k=pidx[i];loc=plan['maps'][str(i)]['centers'];assert np.array_equal(px.reshape(16,49)[:,24],confirm['pixels'][k,loc])
        delta=abs(p.reshape(16,49,2)[:,24]-confirm['c92'][k,loc]);diff.append(delta)
        Y=np.stack([v[i].ravel()[px] for v in yy],1).astype('float32');W=np.stack([np.unpackbits(v[i])[px] for v in ww],1).astype(bool)
        B=base[dev.index(i)].reshape(2,-1)[:,px].T
        changed.append(np.count_nonzero(((Y-np.maximum(B,p)>=5)&W).reshape(16,49,2)[:,24]!=confirm['weak'][k,loc]))
        vals.append(p);truth.append(Y);m.append(W);b.append(B);terrain.append(rows[i]['terrain_id']);reg.append(plan['maps'][str(i)]['arm'])
    diff=np.concatenate(diff);summary=dict(center_pixels=len(diff),max_abs_replay_dB=np.max(diff,0).tolist(),p999=np.quantile(diff,.999,axis=0).tolist(),changes_to_joint_event=int(np.sum(changed)))
    assert np.quantile(diff,.99)<.1 and np.sum(changed)<15,summary
    p,y,w,b=map(np.array,(vals,truth,m,b));weak=(y>=np.maximum(b,p)+5)&w
    np.savez_compressed(A/'patch_target.npz',ids=f['ids'],pixels=f['pixels'],terrain=terrain,groups=np.array(['confirm']*len(terrain)),y=y,valid=w,c82=b,c92=p,weak=weak,arm=reg)
    (A/'patch_replay.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
