"""Preselect independent nearby-pixel tests. This script opens H/TX geometry and no PL or masks."""
import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
def main():
    z=np.load(A/'confirm_geo.npz');ids=z['ids'];px=z['pixels'];sheet=z['sheet'];rng=np.random.default_rng(93241)
    sel=np.sort(rng.choice(len(ids),120,replace=False));maps={}
    for j in sel:
        i=int(ids[j]);rng=np.random.default_rng([931205,i]);y,x=np.divmod(px[j],256)
        # Uniform centers independent of propagation, followed by centers selected by input geometry alone.
        interior=(y>4)&(y<251)&(x>4)&(x<251)
        order=np.flatnonzero(interior);rng.shuffle(order)
        risk=sheet[j,:,3]+.5*sheet[j,:,5]+.2*sheet[j,:,2].clip(0,3)
        risk_order=order[np.argsort(-risk[order],kind='stable')];chosen=[];arm=[]
        for name,seq in [('uniform',order),('grazing',risk_order)]:
            count=0
            for k in seq:
                if all(max(abs(y[k]-y[c]),abs(x[k]-x[c]))>=9 for c in chosen):
                    chosen.append(int(k));arm.append(name);count+=1
                    if count==8:break
            assert count==8
        oy,ox=np.indices((7,7));oy=oy.reshape(-1)-3;ox=ox.reshape(-1)-3
        patch=(y[chosen,None]+oy)*256+x[chosen,None]+ox
        assert len(np.unique(patch))==784 and np.array_equal(patch[:,24],px[j,chosen])
        maps[str(i)]=dict(centers=chosen,arm=arm,px=patch.ravel().tolist())
    plan=dict(terrain_indices=ids[sel].tolist(),maps=maps,seed=93241,purpose='120 of 484 confirm terrains, 8 random + 8 geometry-only grazing centers/terrain, 7x7 nonoverlapping; chosen without label, mask or predictions. Uniform and enriched results separate.')
    (A/'patches.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
if __name__=='__main__':main()
