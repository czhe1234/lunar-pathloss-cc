"""Compile constants for candidate centroid-to-raster interpolation using coordinates only."""
from pathlib import Path
import sys,numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.c66_lattice import Mesh
for anti in [0,1]:
    m=Mesh(1,anti)
    w=m.weights;ids=m.neighbors.copy();ok=w.sum(1)>.5
    axis=np.linspace(1/3,255-1/3,256);qx,qy=np.meshgrid(axis,axis)
    xy=np.stack(np.meshgrid(np.arange(256),np.arange(256)),axis=-1).reshape(-1,2)
    _,near=cKDTree(xy[m.faces].mean(1)).query(np.stack((qx.ravel()[~ok],qy.ravel()[~ok]),1))
    ids[~ok]=near[:,None];w[~ok]=[1,0,0]
    np.savez_compressed(R/f'src/c70_centroid_lookup{anti}.npz',ids=ids.astype('int32'),weights=w.astype('float32'))
