"""True terrain-cell topology and observation stencil, in float32 world coordinates."""
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay,cKDTree
R=Path(__file__).resolve().parents[1];A=R/'audit/c88_mesh'
def ordered_faces(diag):
    m=len(diag);a=(np.arange(m)[:,None]*(m+1)+np.arange(m)).ravel()
    b=a+1;c=a+m+1;d=c+1;s=diag.ravel().astype(bool)
    return np.concatenate([np.stack([a,b,np.where(s,c,d)],1),np.stack([np.where(s,d,a),np.where(s,c,d),np.where(s,b,c)],1)])
def vertex_xy(n):
    t=np.linspace(-128,128,n,dtype=np.float32);x,y=np.meshgrid(t,-t)
    return np.stack([x.ravel(),y.ravel()],1)
def diagonals(n,mode):
    m=n-1
    if mode in ('fixed','opposite'):return np.full((m,m),mode=='opposite',np.uint8)
    if mode=='checker':return (np.indices((m,m)).sum(0)%2).astype(np.uint8)
    tri=Delaunay(vertex_xy(n));f=np.sort(tri.simplices,axis=1)
    cell=(f[:,0]//n)*m+f[:,0]%n
    # A main-diagonal square has its NW and SE vertices together; an anti-diagonal
    # square has a triangle at NW that contains both the NE and SW vertices.
    anti=(f[:,1]==f[:,0]+1)&(f[:,2]==f[:,0]+n)
    out=np.zeros(m*m,np.uint8);out[cell[anti]]=1;out=out.reshape(m,m)
    assert .15<out.mean()<.85
    return np.roll(out,(37,59),(0,1)).copy() if mode=='roll' else out
def project(n,diag):
    xy=vertex_xy(n);fc=ordered_faces(diag)
    centers=xy[fc].mean(1);tri=Delaunay(centers)
    u=np.linspace(centers[:,0].min(),centers[:,0].max(),256)
    v=np.linspace(centers[:,1].max(),centers[:,1].min(),256)
    px,py=np.meshgrid(u,v);points=np.stack([px.ravel(),py.ravel()],1)
    si=tri.find_simplex(points);tf=tri.transform[si];w=np.einsum('nij,nj->ni',tf[:,:2],points-tf[:,2])
    ids=tri.simplices[si].copy();w=np.column_stack([w,1-w.sum(1)])
    bad=si<0
    if bad.any():
        near=cKDTree(centers).query(points[bad])[1];ids[bad]=near[:,None];w[bad]=[1,0,0]
    assert np.max(abs(w.sum(1)-1))<1e-6
    return ids.astype(np.int32),w.astype(np.float32)
def main():
    out=A/'tables';out.mkdir(exist_ok=True)
    for n,mode in [(256,'fixed'),(256,'opposite'),(256,'checker'),(256,'delaunay'),(256,'roll'),(128,'fixed'),(192,'fixed'),(384,'fixed')]:
        p=out/f'{n}_{mode}.npz'
        if p.exists():continue
        dg=diagonals(n,mode)
        if n==256 and mode in ('fixed','opposite'):
            v=np.load(R/f'src/c70_centroid_lookup{int(mode=="opposite")}.npz');ids,w=v['ids'].astype(np.int32),v['weights'].astype(np.float32)
        else:ids,w=project(n,dg)
        np.savez(p,diag=dg,ids=ids,weights=w)
        print(p.name,round(float(dg.mean()),4),flush=True)
if __name__=='__main__':main()
