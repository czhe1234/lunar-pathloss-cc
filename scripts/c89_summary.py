"""Paired terrain intervals and dual-frequency residual on exactly the same confirmation pixels."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c89_survival';C=R/'audit/c88_mesh'
def boot(s,t):
    u,inv=np.unique(t,return_inverse=True);a=np.zeros((len(u),3));np.add.at(a,inv,s)
    idx=np.random.default_rng(89821).integers(len(a),size=(2000,len(a)));z=a[idx].sum(1)
    gains=np.sqrt(z[:,0]/z[:,2])-np.sqrt(z[:,1]/z[:,2])
    return np.quantile(gains,[.025,.975]).tolist()
def main():
    out={'c88':{},'c89':{}}
    for b in (0,1):
        q=np.load(C/f'registered_confirm_{b}.npz');v=q['sse_count']
        out['c88'][b]=dict(base=np.sqrt(v[:,0].sum()/v[:,-1].sum()),registered=np.sqrt(v[:,2].sum()/v[:,-1].sum()),
              gain95=boot(v[:,[0,2,-1]],q['terrain']))
        rs=np.load(A/f'real_b{b}/paired.npz')
        for name in ('continuum','placebo','real'):
            f=A/f'{name}_b{b}';p=np.load(f/'paired.npz');u=p['baseline'][:,0];v=p['candidate'][:,0]
            obj=dict(gain95=boot(np.stack([u[:,0],v[:,0],v[:,1]],1),p['terrain']),
                real_over_this95=boot(np.stack([v[:,0],rs['candidate'][:,0,0],v[:,1]],1),p['terrain']))
            obj.update({k:json.load(open(f/'result.json'))[k] for k in ('base','candidate','gain')})
            out['c89'][name+str(b)]=obj
    mu=json.load(open(C/'base_index.json'));inds=np.flatnonzero(np.array(mu['groups'])=='confirm')
    ref=np.load(R/'audit/c78_clear/train_reference.npy',mmap_mode='r');p=np.load(C/'base.npy',mmap_mode='r')
    pix=np.load(R/'audit/c78_clear/train_panel.npz')['pixels'];d=[np.load(A/f'real_b{b}/correction.npy',mmap_mode='r') for b in (0,1)]
    q=np.zeros(3)
    for j,i in enumerate(inds):
        raw=ref[i];w=(raw[:,4]>.5)&(raw[:,5]>.5)&(raw[:,6]<.01)
        y=raw[:,1]-raw[:,0];p0=p[i].reshape(2,-1)[:,pix[i]].astype('float64')
        er=p0[1]-p0[0]-y;de=d[1][j].astype('float64')-d[0][j].astype('float64')
        q+=[w.sum(),(er[w]**2).sum(),((er+de)[w]**2).sum()]
    out['contrast_shadow_joint']=dict(count=int(q[0]),base=float(np.sqrt(q[1]/q[0])),real=float(np.sqrt(q[2]/q[0])))
    r=json.load(open(A/'real_b1/result.json'))['shadow_joint_branch_summary'];out['mean_predicted_transmission_face_nonzero_proxy']=[r[1]/r[0],r[2]/r[0]]
    (A/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(out['contrast_shadow_joint'],out['mean_predicted_transmission_face_nonzero_proxy'],flush=True)
if __name__=='__main__':main()
