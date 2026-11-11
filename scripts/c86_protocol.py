"""Define the true 4-billion-ray sampling test without reading any new label."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'audit/c79_receiver';C=R/'audit/c78_clear'
rows=json.load(open(A/'rt_confirm0/complete.json'))['records']
g={}
for d in C.glob('train_geom_*of*'):
    v=np.load(d/'geometry.npy',mmap_mode='r');g.update({i:(v,j) for j,i in enumerate(np.load(d/'positions.npy'))})
rng=np.random.default_rng(86851);rng.shuffle(rows);chosen=[]
for r in rows[:40]:
    v,j=g[r['pos']];x=v[j];allowed=(x[:,15]>.5)&(x[:,18]>=65)
    cnt=np.digitize(x[:,20],[20,100,400]);ix=[]
    for h in range(4):
        opt=np.flatnonzero(allowed&(cnt==h));ix.extend(rng.choice(opt,min(8,len(opt)),replace=False).tolist())
    if ix:chosen.append(dict(r,locations=sorted(ix)))
out=dict(question='If higher launch density and real repeated measurement-surface intersections are modeled, does the registered H/TX scene localize individual weak official pixels?',
    scope='Official TRAIN confirmation terrains only. 40 of 64 preselected C79 direct-solver maps, omit those without eligible locations; choose uniform within four input-only count strata on the wide visibility certificate and >=65 m. Intentional high-risk sampling. No fitted network.',
    launch='Use original Sionna RT 1.1.0 and its mesh/measurement intersections. Enumerate Fibonacci directions whose initial ray can reach each target triangle plus margin. Correct estimator for each sparse call by sample_count/N. Compare unrotated and +0.013 rad emission phases with the same scene. Validate sparse sampling against previously recorded dense 4M full-sphere output, and recompute 4B under enlarged caps on a preset subset.',
    controls='4M central aperture must reproduce the complete Sionna4M direct field on those very pixels before interpreting 4B. Use identical points to compare launch phases. Descriptive diagnostic, not a contest score or proof of missing information.',
    maps=chosen)
(A/'c86_protocol.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(chosen),sum(len(s['locations']) for s in chosen))
