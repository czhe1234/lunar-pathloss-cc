"""Isolate output query mapping with stored original per-cell powers. No solver or fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment';P=R/'audit/c93_joint_miss'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','extension']);args=ap.parse_args()
    p=A/'plan.json';plan=json.loads(p.read_text())
    variants={f'pixel_{axis}_{v:+g}':{'pixel':[v,0] if axis=='x' else [0,v]} for axis in ('x','y') for v in (-1,-.5,-.25,.25,.5,1)}
    variants.update({f'pixel_{s}':{'pixel':s} for s in ('vertices','integer','half')})
    plan['configs'].update(variants);p.write_text(json.dumps(plan,indent=2)+'\n')
    data={}
    for d in P.glob('native[0-9]'):
        for z in d.glob('*.npz'):
            if '.partial.' in z.name:continue
            a=np.load(z);data[int(a['index'])]={k:a[k] for k in ('pixel','native_ids','native_inv_square')}
    x=np.linspace(-128,128,256,dtype=np.float32);xx,yy=np.meshgrid(x,-x);xy=np.stack([xx.ravel(),yy.ravel()],1)
    a=np.arange(65536).reshape(256,256)[:-1,:-1].ravel()
    ff=np.concatenate([np.stack([a,a+1,a+257],1),np.stack([a,a+257,a+256],1)])
    centers=xy[ff].mean(1);tri=Delaunay(centers)
    qx,qy=np.meshgrid(np.linspace(centers[:,0].min(),centers[:,0].max(),256),np.linspace(centers[:,1].max(),centers[:,1].min(),256))
    orig=np.stack([qx.ravel(),qy.ravel()],1).astype('float64')
    all_ids=np.load(R/'src/c70_centroid_lookup0.npz')
    for name,cfg in variants.items():
        spec=cfg['pixel'];q=orig.copy()
        if isinstance(spec,list):q+=np.array(spec)[None,:]
        elif spec=='vertices':
            qx,qy=np.meshgrid(np.linspace(-128,128,256),np.linspace(128,-128,256));q=np.stack([qx.ravel(),qy.ravel()],1)
        elif spec=='integer':
            qx,qy=np.meshgrid(np.arange(256)-128,127-np.arange(256));q=np.stack([qx.ravel(),qy.ravel()],1)
        elif spec=='half':
            qx,qy=np.meshgrid(np.arange(256)-127.5,127.5-np.arange(256));q=np.stack([qx.ravel(),qy.ravel()],1)
        ss=tri.find_simplex(q);valid=ss>=0;ss=ss.clip(0);tt=tri.transform[ss];w=np.einsum('nij,nj->ni',tt[:,:2],q-tt[:,2])
        w=np.column_stack([w,1-w.sum(1)]);ids=tri.simplices[ss]
        for e in plan[args.stage]:
            z=data[e['index']];px=z['pixel'];pool=np.full(len(ff),np.nan)
            for native_id,v in zip(z['native_ids'].ravel(),z['native_inv_square'][0].ravel()):
                if np.isfinite(pool[native_id]):assert pool[native_id]==v
                pool[native_id]=v
            ids3=ids[px];ws=w[px];native=pool[ids3]
            available=np.all(np.isfinite(native)|(ws<=1e-7),1)&valid[px]
            native=np.nan_to_num(native,nan=0.)
            dest=A/args.stage/f"{e['index']}_{name}.npz"
            np.savez(dest,pixel=px,native=native,weights=ws,ids=ids3,average=(native*ws).sum(1),valid=available,source='old original-cell direct powers, altered output query only')
    (A/f'pixel_{args.stage}_done.json').write_text(json.dumps(dict(maps=len(plan[args.stage]),configs=list(variants),note='Common finite pool support required for fair evaluation; confirm needs new solver donors.'))+'\n')
    print(args.stage,len(plan[args.stage]),len(variants))
if __name__=='__main__':main()
