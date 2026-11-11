"""Recover direction-count features on two source-defined, unrecovered candidate meshes."""
import ctypes
import numpy as np
from scipy.spatial import Delaunay
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
library = ctypes.CDLL(str(ROOT/'src/libc66_lattice.so'))
ptr = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
library.count_faces.argtypes = [ptr,ctypes.c_int,ctypes.c_longlong,ctypes.c_double,ptr]
library.count_faces.restype = None
def counts(vertices, rays, cutoff=96):
    verts = np.ascontiguousarray(vertices,dtype=np.float64)
    result = np.empty((len(verts),3))
    library.count_faces(verts,len(verts),int(rays),float(cutoff),result)
    return result

class Mesh:
    def __init__(self, physical_grid, antidiagonal):
        axis = (np.linspace(-128,128,256,dtype=np.float32) if physical_grid
                else np.arange(256,dtype=np.float32)-128)
        xx,yy=np.meshgrid(axis,-axis)
        self.xy=np.column_stack((xx.ravel(),yy.ravel()))
        a=np.arange(65536).reshape(256,256)[:-1,:-1].ravel()
        triangles=([a,a+1,a+256],[a+257,a+256,a+1]) if antidiagonal else ([a,a+1,a+257],[a,a+257,a+256])
        self.faces=np.concatenate([np.column_stack(t) for t in triangles])
        centers=self.xy[self.faces].mean(1)
        triangulation=Delaunay(centers)
        if physical_grid:
            x,y=np.meshgrid(np.linspace(centers[:,0].min(),centers[:,0].max(),256),
                            np.linspace(centers[:,1].max(),centers[:,1].min(),256))
        else:x,y=xx,yy
        points=np.column_stack((x.ravel(),y.ravel()))
        simplex=triangulation.find_simplex(points)
        transforms=triangulation.transform[simplex]
        weights=np.einsum('nij,nj->ni',transforms[:,:2],points-transforms[:,2])
        self.weights=np.column_stack((weights,1-weights.sum(1)))
        self.weights[simplex<0]=0
        self.neighbors=triangulation.simplices[simplex]
        self.physical=physical_grid
    def vertices(self,height,row,col,pixels,nominal=False):
        neighbors=self.neighbors[pixels]
        ids,inv=np.unique(neighbors.ravel(),return_inverse=True)
        face=self.faces[ids]
        coords=np.concatenate((self.xy[face],(height.ravel()[face]+1)[...,None]),axis=-1).astype(np.float64)
        z=height[row,col] if nominal else np.floor(height[row,col])
        tx=[col-128.,(127. if self.physical else 128.)-row,z+3.]
        return coords-np.asarray(tx),inv.reshape(neighbors.shape),self.weights[pixels]
