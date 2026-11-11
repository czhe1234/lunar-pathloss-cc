"""Preselect direct replay windows from H/TX and two independent geometry-only controls."""
import csv,json
import numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'audit/c93_joint_miss'
f=np.load(A/'patch_features.npz');ns=f['names'].tolist();v=f['x'];ids=f['ids'];pixels=f['pixels']
pub=json.load(open(A/'patches.json'));rows=list(csv.DictReader(open(R.parent/'data/train_index.csv')))
rng=np.random.default_rng(93899);candidates=[]
for j,i in enumerate(ids):
    h=v[j].reshape(16,49,-1)
    d=np.expm1(h[:,:,ns.index('range')]);clear=h[:,:,ns.index('vis')]>.99
    ok=(d.min(1)>=65)&(clear.mean(1)>=.80)
    arms=np.array(pub['maps'][str(i)]['arm']);local=np.random.default_rng([93899,int(i)])
    sel=[]
    for m in ('uniform','grazing'):
        use=np.flatnonzero(ok&(arms==m))
        if len(use):sel.append(int(local.choice(use)))
    if sel:candidates.append(dict(index=int(i),terrain=rows[i]['terrain_id'],patch_map=j,windows=sel,arms=arms[sel].tolist()))
order=rng.permutation(len(candidates))[:60];maps=[candidates[n] for n in order]
out=dict(count=len(maps),eligible=len(candidates),windows=sum(len(x['windows']) for x in maps),maps=maps,
     selection='No measured power or model outputs. 7x7 patches, all cells >=65 m, >=80% geometrically clear, <=1 random and 1 grazing window per terrain, 60 seeded distinct terrains. Nominal lattice and +/- azimuth shifts, same candidate scene. Every fifth terrain repeats nominal at 2.4 m angular padding as a support check. No training or fitted fusion.')
(A/'exact_windows.json').write_text(json.dumps(out,indent=2)+'\n');print(len(maps),out['windows'],out['eligible'])
