"""Build plausible observation templates from native receiver centroids, never from PL values."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
R=Path(__file__).resolve().parents[1];A=R/'audit/c94_measurement'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('tag');args=ap.parse_args()
    x=np.linspace(-128,128,256,dtype=np.float32);xx,yy=np.meshgrid(x,-x);vertices=np.stack([xx.ravel(),yy.ravel()],1)
    a=np.arange(65536).reshape(256,256)[:-1,:-1].ravel()
    faces=np.concatenate([np.stack([a,a+1,a+257],1),np.stack([a,a+257,a+256],1)])
    mode={'numpy_mean':vertices[faces].mean(1),'sionna_centers':np.load(A/'sionna_centers.npy'),
          'exact_mean':vertices[faces].astype('float64').mean(1)}
    out=A/('templates_'+args.tag);out.mkdir(exist_ok=True);base=np.load(R/'src/c70_centroid_lookup0.npz')
    rows={}
    for name,center in mode.items():
        for rev in (False,True):
            pts=center[::-1] if rev else center;tri=Delaunay(pts)
            for grid_dtype in ('default','f64','f32'):
                kind=None if grid_dtype=='default' else np.float64 if grid_dtype=='f64' else np.float32
                x=np.linspace(center[:,0].min(),center[:,0].max(),256,dtype=kind)
                y=np.linspace(center[:,1].max(),center[:,1].min(),256,dtype=kind)
                gx,gy=np.meshgrid(x,y);q=np.stack([gx.ravel(),gy.ravel()],1);s=tri.find_simplex(q)
                b=tri.transform[s];w=np.einsum('nij,nj->ni',b[:,:2],q-b[:,2]);w=np.column_stack([w,1-w.sum(1)])
                ix=tri.simplices[s].copy()
                if rev:ix=len(center)-1-ix
                # The first/last lattice margins are outside a handful of centroid hull simplices.
                bad=s<0;ix[bad]=base['ids'][bad];w[bad]=base['weights'][bad]
                key=name+('_reverse' if rev else '')+'_'+grid_dtype
                np.savez_compressed(out/(key+'.npz'),ids=ix.astype('int32'),weights=w.astype('float64'),query=q)
                current=(abs(w)<1e-6).mean()
                rows[key]=dict(numpy=np.__version__,grid_dtype=str(q.dtype),changed_donor_rows=int((np.sort(ix,axis=1)!=np.sort(base['ids'],axis=1)).any(1).sum()),zero_weight_fraction=float(current),outside=int(bad.sum()))
    (out/'index.json').write_text(json.dumps(rows,indent=2)+'\n');print(args.tag,np.__version__,{k:v['changed_donor_rows'] for k,v in rows.items()})
if __name__=='__main__':main()
