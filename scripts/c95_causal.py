"""Compare rigid-coordinate and receiver-advance controls; no labels used for this invariance test."""
import argparse,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','extension','confirm']);args=ap.parse_args()
    plan=json.loads((A/'plan.json').read_text());out={}
    pairs=[('base','step_formula'),('base','origin_xy128'),('base','origin_z128'),('base','origin_z1024'),
           ('base','origin_xy128_localstep'),('base','origin_z1024_localstep'),
           ('first_base','first_xy128'),('first_base','first_z1024')]
    for c0,c1 in pairs:
        xs=[];ys=[];nn=[]
        for e in plan[args.stage]:
            p=A/args.stage/f"{e['index']}_{c0}.npz";q=A/args.stage/f"{e['index']}_{c1}.npz"
            if not p.exists() or not q.exists():continue
            a,b=np.load(p),np.load(q);assert np.array_equal(a['pixel'],b['pixel'])
            xs.append(a['average']);ys.append(b['average']);nn.append(e['index'])
        if not xs:continue
        x,y=np.concatenate(xs),np.concatenate(ys);both=(x>0)&(y>0);changedzero=(x>0)!=(y>0)
        db=abs(10*np.log10(y[both]/x[both]))
        rel=abs(y-x)/np.maximum(x,1e-10)
        out[c0+'__'+c1]=dict(maps=len(nn),points=len(x),changed_zero=int(changedzero.sum()),positive_both=int(both.sum()),
            change5percent=int((rel>.05).sum()),change1dB_finite=int((db>1).sum()),change3dB_finite=int((db>3).sum()),
            finite_dB_quantiles=np.quantile(db,[0,.5,.9,.95,.99,1]).tolist() if len(db) else [],max_abs=float(abs(y-x).max()))
    (A/f'{args.stage}_invariance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
