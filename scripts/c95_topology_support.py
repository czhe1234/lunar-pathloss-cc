"""Constrain centroid-export hull using development masks and coordinate/vertex-order hypotheses only."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import Delaunay,ConvexHull
R=Path(__file__).resolve().parents[1];A=R/'audit/c95_alignment'
actual=json.loads((A/'mask_export_support.json').read_text());records=[]
for grid in ('centered256','positive256','positive255'):
    axis=np.linspace(-128,128,256,dtype=np.float32) if grid=='centered256' else (np.linspace(0,256,256,dtype=np.float32) if grid=='positive256' else np.arange(256,dtype=np.float32))
    xx,yy=np.meshgrid(axis,-axis if grid=='centered256' else axis[::-1]);xy=np.stack([xx.ravel(),yy.ravel()],1)
    for order in range(8):
        a=np.arange(65536).reshape(256,256)
        if order&4:a=a.T
        if order&1:a=a[::-1]
        if order&2:a=a[:,::-1]
        seq=a.ravel();raw=Delaunay(xy[seq]);faces=seq[raw.simplices]
        c=xy[faces].astype('float64').mean(1);hull=ConvexHull(c)
        corn=np.array([[c[:,0].min(),c[:,1].max()],[c[:,0].max(),c[:,1].max()],[c[:,0].min(),c[:,1].min()],[c[:,0].max(),c[:,1].min()]])
        support=np.max(corn@hull.equations[:,:2].T+hull.equations[:,2],axis=1)<=1e-6
        invalid_valid=sum(actual['415']['corner_valid'][j] for j in range(4) if not support[j])
        records.append(dict(grid=grid,order=order,supported_corners=support.astype(int).tolist(),published_valid_but_outside=invalid_valid))
(A/'topology_support.json').write_text(json.dumps(records,indent=2)+'\n')
for v in records:print(v)
