import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.c66_lattice import counts
rng=np.random.default_rng(123)
centers=rng.uniform([-9,-9,-4],[9,9,4],(150,1,3))
tris=centers+rng.normal(0,.6,(150,3,3))
N=65001
i=np.arange(N,dtype=np.float64)
phi=(2*np.pi*np.remainder(i/((1+np.sqrt(5))/2),1).astype(np.float32)).astype(np.float32)
z=1-2*(i/(N-1)).astype(np.float32);r=np.sqrt(1-z*z)
dirs=np.column_stack((r*np.cos(phi),r*np.sin(phi),z))
got=counts(tris,N,1e7);bad=[]
for idx,v in enumerate(tris):
    if got[idx,2]<0:continue
    sides=np.cross(v,v[[1,2,0]])
    dots=(dirs@sides.T)*np.sign(sides[0]@v[2])
    exact=(dots>=0).all(1).sum()
    if exact!=got[idx,2]:bad.append((idx,int(exact),float(got[idx,2])))
assert not bad,bad[:8]
print('Brute-force direction counts match for',int((got[:,2]>=0).sum()),'triangles.')
