"""Generate receiver-cell counting diagnostics from inputs, on a fixed FIT/DEV panel."""
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c66_lattice import Mesh, counts
A=ROOT/'audit/c66_transport'
PANEL=json.loads((A/'panels.json').read_text())
rows=list(csv.DictReader((ROOT.parent/'data/train_index.csv').open()))
height=np.load(ROOT.parent/'data/train_hm.npy',mmap_mode='r')
rng=np.random.default_rng(20261104)
training=rng.choice(PANEL['fit_first'],64,replace=False)
confirmation=rng.choice(PANEL['dev_first'],64,replace=False)
maps=np.concatenate((training,confirmation))
pixels=rng.choice(65536, (len(maps),512),replace=True)
sizes=[1000000,4000000,10000000,40000000,100000000,400000000,1000000000,4000000000]
shape=(2,2,len(sizes),len(maps),pixels.shape[1],3)
measures=np.lib.format.open_memmap(A/'lattice_features.npy',mode='w+',dtype='float32',shape=shape)
start=time.monotonic()
for phys in [0,1]:
 for diagonal in [0,1]:
    mesh=Mesh(phys,diagonal)
    for j,idx in enumerate(maps):
        row=rows[idx]
        verts,inv,w=mesh.vertices(height[int(row['hm_row'])],int(row['tx_row']),int(row['tx_col']),pixels[j])
        for k,n in enumerate(sizes):
            raw=counts(verts,n)
            ratio=raw[:,0][inv]
            # Two plausible observation domains and one expectation-only diagnostic.
            measures[phys,diagonal,k,j,:,0]=(ratio*w).sum(1)
            measures[phys,diagonal,k,j,:,1]=np.sum(w*np.log10(.03+ratio),axis=1)-np.log10(1.03)
            measures[phys,diagonal,k,j,:,2]=np.sum(w/np.maximum(raw[:,1][inv],.01),axis=1)
            measures[phys,diagonal,k,j,w.sum(1)==0]=[1,0,0]
    measures.flush()
    print(json.dumps({'physical_grid':phys,'anti_diagonal':diagonal,'seconds':time.monotonic()-start}),flush=True)
np.savez(A/'lattice_panel.npz',maps=maps,pixels=pixels,rays=sizes)
(A/'lattice_features_complete.json').write_text(json.dumps({'maps':len(maps),'FIT':len(training),'DEV':len(confirmation),'pixels_per_map':512,'seconds':time.monotonic()-start,'input_only':True,'geometry':'Both 1m/public-index model and 256m/255interval centroid-bounds model, candidate floor TXz; both diagonals. Ray-count threshold96, above it use continuum. No official VAL.'}))
