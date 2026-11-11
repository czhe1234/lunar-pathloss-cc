"""Held-out TX intervention: can changes in geometry rank the actual changed weak cells? No adjacent label is an input."""
import csv,json
import numpy as np
from pathlib import Path
from c93_geometry import read_compact,terrain_view
from c93_seed_pairs import load
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss';D=R.parent/'data'
def main():
    plan=json.load(open(A/'tx_plan.json'));phys=load('tx');rows=list(csv.DictReader(open(D/'train_index.csv')))
    Y=[np.load(D/f'train_pl{b}.npy',mmap_mode='r') for b in ('415','58')];W=[np.load(D/f'train_mask{b}.npy',mmap_mode='r') for b in ('415','58')]
    H=np.load(D/'train_hm.npy',mmap_mode='r');P=np.load(R/'audit/c88_mesh/base.npy',mmap_mode='r')
    bi={i:j for j,i in enumerate(json.load(open(R/'audit/c88_mesh/base_index.json'))['ids'])}
    compact=read_compact();blank=np.zeros((256,256,10),dtype='float16');table={}
    for i in plan['ids']:
        row=rows[i];tr,tc=int(row['tx_row']),int(row['tx_col']);pix=np.array(plan['pixels'][row['terrain_id']]);h=H[int(row['hm_row'])]
        c,k=compact[i];y=np.stack([v[i].ravel()[pix] for v in Y],1);w=np.stack([np.unpackbits(v[i])[pix] for v in W],1).astype(bool)
        p=P[bi[i]].reshape(2,-1)[:,pix].T
        hit=(y>=np.maximum(p,phys[i])+5)&w
        risks=[]
        for rot in (False,True):
            v,f=terrain_view(h,tr,tc,pix,c[k],blank,rot=rot)
            risks.append((v['graze']+v['shot']+v['small_mass'])/3)
        table[i]=(risks,y,w,hit,v['state']*2)
    result={}
    for typ in ('near','far'):
        pairs=[p for p in plan['pairs'] if p['group']==typ];names=np.unique([p['terrain'] for p in pairs])
        ctr=np.zeros((len(names),2,3,4))
        for p in pairs:
            ua,ya,wa,ha,sa=table[p['i']];ub,yb,wb,hb,sb=table[p['j']];k=list(names).index(p['terrain'])
            for b in (0,1):
                for z in (0,1,2):
                    m=wa[:,b]&wb[:,b]&(sa==sb)&((sa==0) if z==1 else (sa==2) if z==2 else True)&(ha[:,b]!=hb[:,b])
                    sign=np.where(hb[:,b],1.,-1.);ctr[k,b,z,0]+=m.sum()
                    for r in (0,1):
                        delta=(ub[r]-ua[r])[m]*sign[m];ctr[k,b,z,r+1]+=(.5+.5*np.sign(delta)).sum()
                    ctr[k,b,z,3]+=(np.abs(ya[:,b]-yb[:,b])[m]).sum(dtype='float64')
        rr={}
        for b in range(2):
            rr[b]={}
            for z,s in enumerate(('all_same_state','same_clear','same_shadow')):
                v=ctr[:,b,z];total=v.sum(0);boot=v[np.random.default_rng(93779).integers(len(v),size=(2000,len(v)))].sum(1)
                use=boot[:,0]>0;pro=boot[use,1:3]/boot[use,0:1]
                rr[b][s]=dict(n_changed_positions=int(total[0]),true_direction=float(total[1]/total[0]),rotated_direction=float(total[2]/total[0]),
                    true_direction95=np.quantile(pro[:,0],[.025,.975]).tolist(),true_minus_rot95=np.quantile(pro[:,0]-pro[:,1],[.025,.975]).tolist(),power_MAE=float(total[3]/total[0]))
        result[typ]=dict(terrains=len(names),pairs=len(pairs),results=rr)
    (A/'tx_change.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result)
if __name__=='__main__':main()
